use crate::passthrough::compile_program;
use crate::room_scanner::{
    classify_ar_plane, infer_floor_y, AabbCollider, PlaneMarker, SurfaceKind,
};
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::collections::HashMap;
use std::mem::size_of;

pub const CHUNK_DIM: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM; // 4096 voxels/chunk
pub const CELL_SIZE: f32 = 0.025; // 2.5cm ultra-fine base grid -> 0.40m cube/chunk (close-up bumpy detail!)
pub const TSDF_TRUNC: f32 = CELL_SIZE * 3.0; // ±7.5cm narrow band (eliminates carving dead-zone!)
pub const CLOSE_DETAIL_DIST_CM: u8 = 95; // <= 0.95m camera distance unlocks ultra-fine 2.5cm triangles; > 0.95m merges 2x2 (5cm)
pub const MAX_ACTIVE_CHUNKS: usize = 320; // ~5.1 MB TSDF RAM cap (fits effortlessly in 6-8 GB phone RAM)
pub const MAX_CANONICAL_PLANES: usize = 24;
pub const MAX_POLY_VERTS: usize = 24;
pub const MAX_MESH_VERTICES: usize = 90_000; // 30,000 triangles sorted nearest-first (~4.68 MB VBO)
pub const MAX_RENDER_RADIUS_M: f32 = 5.5;

pub const MESH_MIN_WEIGHT: u8 = 24; // Locks in 2 frames close/high-conf or 3 frames smooth; rejects 1-frame noise
pub const LOCK_CHANGE_CONFIRM: u8 = 3; // True Lock mode requires 3 consecutive frames (~1.5s) before mutating TSDF

const TRI_MESH_VERT: &str = r#"#version 300 es
precision highp float;
in vec3 a_pos;
in vec3 a_normal;
in vec3 a_bary;
in vec4 a_color;
uniform mat4 u_mvp;
out vec3 v_normal;
out vec3 v_bary;
out vec4 v_color;
void main() {
    v_normal = a_normal;
    v_bary = a_bary;
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const TRI_MESH_FRAG: &str = r#"#version 300 es
precision mediump float;
in vec3 v_normal;
in vec3 v_bary;
in vec4 v_color;
out vec4 frag_color;
void main() {
    vec3 d = fwidth(v_bary);
    vec3 a3 = smoothstep(vec3(0.0), max(d * 1.35, vec3(0.0018)), v_bary);
    float edge_factor = 1.0 - min(min(a3.x, a3.y), a3.z);

    vec3 n = normalize(v_normal);
    float shade = 0.54 + 0.46 * clamp(dot(n, normalize(vec3(0.32, 0.86, 0.40))) * 0.5 + 0.5, 0.0, 1.0);
    vec3 fill_rgb = v_color.rgb * shade;
    vec3 wire_rgb = mix(v_color.rgb, vec3(1.0), 0.48);

    float fill_a = v_color.a;
    float wire_a = clamp(v_color.a * 2.5 + 0.46, 0.50, 0.90);

    vec3 out_rgb = mix(fill_rgb, wire_rgb, edge_factor);
    float out_a = mix(fill_a, wire_a, edge_factor);
    frag_color = vec4(out_rgb, out_a);
}
"#;

const DEPTH_OCC_VERT: &str = r#"#version 300 es
precision highp float;
in vec3 a_pos;
in vec3 a_normal;
uniform mat4 u_mvp;
uniform float u_normal_bias;
void main() {
    vec3 pushed = a_pos - normalize(a_normal) * u_normal_bias;
    gl_Position = u_mvp * vec4(pushed, 1.0);
}
"#;

const DEPTH_OCC_FRAG: &str = r#"#version 300 es
precision lowp float;
out vec4 frag_color;
void main() {
    frag_color = vec4(0.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TriMeshVertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub bary: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct RawArPlane {
    pub raw_type: i32,
    pub pos: Vec3,
    pub rot: Quat,
    pub extent_x: f32,
    pub extent_z: f32,
    pub polygon_xz: Vec<[f32; 2]>,
}

/// ARCore planes kept silently for invisible floor elevation (`physics_3d.rs`) and reticle hit classification.
/// NEVER rendered visually as giant floor carpets or stuck blue walls!
#[derive(Clone, Debug)]
pub struct CanonicalPlane {
    pub kind: SurfaceKind,
    pub raw_type: i32,
    pub pos: Vec3,
    pub rot: Quat,
    pub normal: Vec3,
    pub polygon_xz: Vec<[f32; 2]>,
    pub extent_x: f32,
    pub extent_z: f32,
    pub area: f32,
    pub weight: f32,
    pub last_seen_gen: u64,
    pub is_explicit_patch: bool,
    pub carve_hits: u16,
    pub detail_hits: u16,
    pub demoted_to_object: bool,
}

impl CanonicalPlane {
    pub fn to_plane_marker(&self) -> PlaneMarker {
        PlaneMarker {
            kind: self.kind,
            pos: self.pos,
            rot: self.rot,
            extent_x: self.extent_x,
            extent_z: self.extent_z,
        }
    }

    pub fn bounding_radius(&self) -> f32 {
        0.5 * (self.extent_x * self.extent_x + self.extent_z * self.extent_z).sqrt()
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct DepthIngestStats {
    pub valid_rays: u32,
    pub changed_rays: u32,
    pub updated_voxels: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct SavedTsdfVoxel {
    pub world_pos: Vec3,
    pub tsdf: i8,
    pub weight: u8,
}

struct VoxelChunk {
    tsdf: Box<[i8; CHUNK_VOLUME]>,        // [-127..127] => [-1.0..+1.0] (+ air, 0 surface, - inside)
    weight: Box<[u8; CHUNK_VOLUME]>,      // Multi-view weight (0..=80, >= MESH_MIN_WEIGHT locked)
    min_dist_cm: Box<[u8; CHUNK_VOLUME]>, // Closest observation distance in cm (0..=255): drives adaptive LOD & close-up override!
    change_hits: Box<[u8; CHUNK_VOLUME]>, // Bit-packed consecutive-frame vote tracker `[5b tag | 3b count]`
    surface_count: u16,
    dirty: bool,
    last_touched_gen: u64,
    detail_tris: Vec<TriMeshVertex>, // High-detail bumpy/curved triangles (2.5cm close-up / 5cm mid-range)
    flat_tris: Vec<TriMeshVertex>,   // Coplanar greedy-merged flat sheets (1 crisp quad per flat face!)
    aabbs: Vec<AabbCollider>,
}

impl VoxelChunk {
    fn new(gen: u64) -> Self {
        Self {
            tsdf: Box::new([127i8; CHUNK_VOLUME]),
            weight: Box::new([0u8; CHUNK_VOLUME]),
            min_dist_cm: Box::new([255u8; CHUNK_VOLUME]),
            change_hits: Box::new([0u8; CHUNK_VOLUME]),
            surface_count: 0,
            dirty: false,
            last_touched_gen: gen,
            detail_tris: Vec::new(),
            flat_tris: Vec::new(),
            aabbs: Vec::new(),
        }
    }

    #[inline]
    fn idx(lx: usize, ly: usize, lz: usize) -> usize {
        (ly * CHUNK_DIM + lz) * CHUNK_DIM + lx
    }

    #[inline]
    fn is_near_surface(tsdf: i8, weight: u8) -> bool {
        weight >= MESH_MIN_WEIGHT && (-102..=26).contains(&tsdf)
    }

    #[inline]
    fn total_tris_len(&self) -> usize {
        self.detail_tris.len() + self.flat_tris.len()
    }

    /// Bit-packed consecutive-frame vote tracker in 1 byte (`[5-bit tick_tag | 3-bit vote_count]`).
    #[inline]
    fn vote_change(&mut self, idx: usize, gen: u64) -> u8 {
        let cur_tag = (gen & 0x1F) as u8;
        let entry = self.change_hits[idx];
        let prev_cnt = entry & 0x07;
        if prev_cnt == 0 {
            self.change_hits[idx] = (cur_tag << 3) | 1;
            return 1;
        }
        let prev_tag = entry >> 3;
        if prev_tag == cur_tag {
            return prev_cnt;
        }
        let delta = cur_tag.wrapping_sub(prev_tag) & 0x1F;
        let next_cnt = if delta <= 2 {
            prev_cnt.saturating_add(1).min(7)
        } else {
            1
        };
        self.change_hits[idx] = (cur_tag << 3) | next_cnt;
        next_cnt
    }

    /// Distance-adaptive TSDF fusion:
    /// - Moving **closer** (`obs_dist_cm + 18 < old_min_dist_cm`) decays stale distant weight so close-up
    ///   high-accuracy depth immediately overwrites old distant errors and refines triangles to `2.5cm`!
    /// - Moving **farther** (`obs_dist_cm > old_min_dist_cm + 35`) down-weights noisy far rays so close-up
    ///   scans stay crisp forever!
    #[inline]
    fn fuse_voxel(
        &mut self,
        idx: usize,
        sdf_norm: f32,
        mut add_w: f32,
        obs_dist_cm: u8,
        is_carve: bool,
        gen: u64,
    ) -> bool {
        let old_t = self.tsdf[idx];
        let old_w = self.weight[idx];
        let old_d = self.min_dist_cm[idx];
        let was_surf = Self::is_near_surface(old_t, old_w);

        let mut old_wf = old_w as f32;
        if is_carve {
            // Instant self-healing: cap old weight aggressively so floating errors vanish in 1-2 frames!
            old_wf = old_wf.min(16.0);
        } else if old_w > 0 && obs_dist_cm < 250 {
            if (obs_dist_cm as u16) + 18 < (old_d as u16) {
                // Much closer view -> override stale distant error cleanly!
                old_wf = old_wf.min(12.0);
                add_w = (add_w * 1.45).min(30.0);
            } else if (obs_dist_cm as u16) > (old_d as u16) + 35 {
                // Distant view looking at an already close-up detailed voxel -> barely perturb it!
                add_w *= 0.25;
            }
        }

        let new_wf = (old_wf + add_w).min(80.0);
        let old_tf = old_t as f32 / 127.0;
        let fused = if old_w == 0 {
            sdf_norm
        } else {
            (old_tf * old_wf + sdf_norm * add_w) / (old_wf + add_w)
        };

        let new_t = (fused * 127.0).round().clamp(-127.0, 127.0) as i8;
        let new_w = new_wf.round().clamp(1.0, 80.0) as u8;

        self.tsdf[idx] = new_t;
        self.weight[idx] = new_w;
        self.change_hits[idx] = 0;
        self.last_touched_gen = gen;

        let mut lod_refined = false;
        if is_carve {
            if new_t > 20 {
                self.min_dist_cm[idx] = 255;
            }
        } else if obs_dist_cm < old_d {
            let crossed_close = old_d > CLOSE_DETAIL_DIST_CM && obs_dist_cm <= CLOSE_DETAIL_DIST_CM;
            self.min_dist_cm[idx] = obs_dist_cm;
            if crossed_close && new_w >= MESH_MIN_WEIGHT {
                lod_refined = true;
            }
        }

        let now_surf = Self::is_near_surface(new_t, new_w);
        if !was_surf && now_surf {
            self.surface_count = self.surface_count.saturating_add(1);
        } else if was_surf && !now_surf {
            self.surface_count = self.surface_count.saturating_sub(1);
        }

        let lock_changed = (old_w < MESH_MIN_WEIGHT) != (new_w < MESH_MIN_WEIGHT);
        let sign_changed = (old_t > 0) != (new_t > 0);
        let shifted = new_w >= MESH_MIN_WEIGHT && (new_t as i16 - old_t as i16).abs() >= 6;
        if lock_changed || sign_changed || shifted || lod_refined {
            self.dirty = true;
            return true;
        }
        false
    }
}

pub struct SpatialMeshMap {
    pub planes: Vec<CanonicalPlane>,
    chunks: HashMap<(i32, i32, i32), VoxelChunk>,
    pub has_dense_depth: bool,
    pub tick_gen: u64,
    pub mesh_gen: u64,
    last_floor_y: Option<f32>,
}

impl Default for SpatialMeshMap {
    fn default() -> Self {
        Self {
            planes: Vec::with_capacity(MAX_CANONICAL_PLANES),
            chunks: HashMap::with_capacity(128),
            has_dense_depth: false,
            tick_gen: 0,
            mesh_gen: 1,
            last_floor_y: None,
        }
    }
}

fn push_mesh_tri(
    out: &mut Vec<TriMeshVertex>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    normal: Vec3,
    color: [f32; 4],
) {
    if out.len() + 3 > MAX_MESH_VERTICES {
        return;
    }
    let n = normal.to_array();
    out.push(TriMeshVertex {
        pos: p0.to_array(),
        normal: n,
        bary: [1.0, 0.0, 0.0],
        color,
    });
    out.push(TriMeshVertex {
        pos: p1.to_array(),
        normal: n,
        bary: [0.0, 1.0, 0.0],
        color,
    });
    out.push(TriMeshVertex {
        pos: p2.to_array(),
        normal: n,
        bary: [0.0, 0.0, 1.0],
        color,
    });
}

fn push_mesh_quad(
    out: &mut Vec<TriMeshVertex>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    normal: Vec3,
    color: [f32; 4],
) {
    push_mesh_tri(out, p0, p1, p2, normal, color);
    push_mesh_tri(out, p0, p2, p3, normal, color);
}

/// Colors adaptive mesh regions distinctly:
/// - **Flat / Simple Merged Surfaces:** clean architectural cool blue/cyan (`alpha = 0.12`)
/// - **Close-Up Ultra-Fine Bumpy Detail (`2.5cm`):** vivid warm amber-gold (`alpha = 0.25`)
/// - **Mid-Range Bumpy Detail (`5.0cm`):** crisp emerald-mint (`alpha = 0.20`)
fn adaptive_region_color(is_flat_merged: bool, is_close_fine: bool, curv: f32, normal: Vec3) -> [f32; 4] {
    if is_flat_merged {
        if normal.y >= 0.78 {
            return [0.22, 0.80, 0.92, 0.14]; // Flat tabletop / horizontal simple surface: clean sky-cyan
        }
        return [0.30, 0.62, 0.98, 0.12]; // Flat wall / box side / cabinet door: clean architectural blue
    }
    if is_close_fine {
        let t = ((curv - 0.08) / 0.24).clamp(0.0, 1.0);
        return [1.00, 0.74 - 0.28 * t, 0.16 + 0.10 * (1.0 - t), 0.25]; // Close-up 2.5cm fine detail: warm gold/amber
    }
    [0.14, 0.90, 0.68, 0.20] // Mid-range 5.0cm bumpy detail: crisp emerald-mint
}

/// Computes a deduplicated CCW 2D Convex Hull using Andrew's Monotone Chain.
pub fn convex_hull_2d(pts: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut clean: Vec<[f32; 2]> = pts
        .iter()
        .copied()
        .filter(|p| p[0].is_finite() && p[1].is_finite())
        .collect();
    if clean.len() < 3 {
        return clean;
    }
    clean.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    clean.dedup_by(|a, b| (a[0] - b[0]).abs() < 0.015 && (a[1] - b[1]).abs() < 0.015);
    if clean.len() < 3 {
        return clean;
    }

    let cross = |o: [f32; 2], a: [f32; 2], b: [f32; 2]| -> f32 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };

    let mut lower: Vec<[f32; 2]> = Vec::with_capacity(clean.len());
    for &p in &clean {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 1e-5 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f32; 2]> = Vec::with_capacity(clean.len());
    for &p in clean.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 1e-5 {
            upper.pop();
        }
        upper.push(p);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.len() > MAX_POLY_VERTS {
        let step = lower.len() as f32 / MAX_POLY_VERTS as f32;
        let mut decimated = Vec::with_capacity(MAX_POLY_VERTS);
        for i in 0..MAX_POLY_VERTS {
            decimated.push(lower[((i as f32) * step) as usize % lower.len()]);
        }
        return decimated;
    }
    lower
}

fn poly_metrics(poly: &[[f32; 2]]) -> ([f32; 2], f32, f32, f32) {
    if poly.is_empty() {
        return ([0.0, 0.0], 0.0, 0.0, 0.0);
    }
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_z = f32::INFINITY;
    let mut max_z = f32::NEG_INFINITY;
    let mut area2 = 0.0f32;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        min_x = min_x.min(a[0]);
        max_x = max_x.max(a[0]);
        min_z = min_z.min(a[1]);
        max_z = max_z.max(a[1]);
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    let center = [(min_x + max_x) * 0.5, (min_z + max_z) * 0.5];
    let ex = (max_x - min_x).max(0.05);
    let ez = (max_z - min_z).max(0.05);
    (center, (area2.abs() * 0.5).max(ex * ez * 0.25), ex, ez)
}

fn rect_fallback_poly(ex: f32, ez: f32) -> Vec<[f32; 2]> {
    let hx = (ex * 0.5).clamp(0.05, 4.0);
    let hz = (ez * 0.5).clamp(0.05, 4.0);
    vec![[-hx, -hz], [hx, -hz], [hx, hz], [-hx, hz]]
}

#[inline]
fn read_depth_mm(depth_bytes: &[u8], x: usize, y: usize, row_stride: usize, pix_stride: usize) -> u16 {
    let off = y * row_stride + x * pix_stride;
    if off + 1 >= depth_bytes.len() {
        return 0;
    }
    u16::from_le_bytes([depth_bytes[off], depth_bytes[off + 1]])
}

#[inline]
fn read_conf_u8(
    cb: &[u8],
    x: usize,
    y: usize,
    dw: usize,
    dh: usize,
    cw: usize,
    ch: usize,
    c_row_stride: usize,
    c_pix_stride: usize,
) -> u8 {
    if cw == 0 || ch == 0 {
        return 0;
    }
    let cx_i = (x * cw / dw).min(cw - 1);
    let cy_i = (y * ch / dh).min(ch - 1);
    let off = cy_i * c_row_stride + cx_i * c_pix_stride;
    if off < cb.len() { cb[off] } else { 0 }
}

#[allow(clippy::too_many_arguments)]
fn max_nearby_conf(
    cb: &[u8],
    x: usize,
    y: usize,
    dw: usize,
    dh: usize,
    cw: usize,
    ch: usize,
    c_row_stride: usize,
    c_pix_stride: usize,
) -> u8 {
    let mut best = 0u8;
    for (dx, dy) in [
        (-2isize, 0isize),
        (2, 0),
        (0, -2),
        (0, 2),
        (-4, 0),
        (4, 0),
        (0, -4),
        (0, 4),
    ] {
        let nx = (x as isize + dx).clamp(0, (dw - 1) as isize) as usize;
        let ny = (y as isize + dy).clamp(0, (dh - 1) as isize) as usize;
        let c = read_conf_u8(cb, nx, ny, dw, dh, cw, ch, c_row_stride, c_pix_stride);
        if c > best {
            best = c;
            if best >= 60 {
                return best;
            }
        }
    }
    best
}

/// Connected-component flood fill inside a chunk (`16×16×16`) that marks isolated floating mid-air
/// specks (`< 7` connected voxels not touching any chunk border) as `false` so floating confetti NEVER renders!
fn compute_non_floater_mask(chunk: &VoxelChunk) -> [bool; CHUNK_VOLUME] {
    let mut keep = [false; CHUNK_VOLUME];
    let mut visited = [false; CHUNK_VOLUME];
    let mut queue = Vec::with_capacity(64);
    let mut cluster = Vec::with_capacity(64);

    for ly in 0..CHUNK_DIM {
        for lz in 0..CHUNK_DIM {
            for lx in 0..CHUNK_DIM {
                let start = VoxelChunk::idx(lx, ly, lz);
                if visited[start]
                    || !VoxelChunk::is_near_surface(chunk.tsdf[start], chunk.weight[start])
                {
                    continue;
                }
                visited[start] = true;
                queue.clear();
                cluster.clear();
                queue.push((lx, ly, lz));
                let mut touches_border = false;

                while let Some((cx, cy, cz)) = queue.pop() {
                    let ci = VoxelChunk::idx(cx, cy, cz);
                    cluster.push(ci);
                    if cx == 0
                        || cy == 0
                        || cz == 0
                        || cx == CHUNK_DIM - 1
                        || cy == CHUNK_DIM - 1
                        || cz == CHUNK_DIM - 1
                    {
                        touches_border = true;
                    }
                    let nbs = [
                        (cx.wrapping_sub(1), cy, cz),
                        (cx + 1, cy, cz),
                        (cx, cy.wrapping_sub(1), cz),
                        (cx, cy + 1, cz),
                        (cx, cy, cz.wrapping_sub(1)),
                        (cx, cy, cz + 1),
                    ];
                    for (nx, ny, nz) in nbs {
                        if nx < CHUNK_DIM && ny < CHUNK_DIM && nz < CHUNK_DIM {
                            let ni = VoxelChunk::idx(nx, ny, nz);
                            if !visited[ni]
                                && VoxelChunk::is_near_surface(chunk.tsdf[ni], chunk.weight[ni])
                            {
                                visited[ni] = true;
                                queue.push((nx, ny, nz));
                            }
                        }
                    }
                }

                // Keep real surfaces (cluster >= 7 voxels OR spanning across chunk border)
                if touches_border || cluster.len() >= 7 {
                    for &ci in &cluster {
                        keep[ci] = true;
                    }
                }
            }
        }
    }
    keep
}

#[derive(Copy, Clone)]
struct CandidateQuad {
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    normal: Vec3,
    curv: f32,
    min_dist_cm: u8,
    is_flat: bool,
    plane_bin: i16,
}

/// Projects merged flat rectangle corners onto the fitted average axis coordinate so coplanar surfaces
/// form a single razor-flat sheet (1 unified quad = 2 triangles instead of dozens of bumpy micro-tris!).
#[allow(clippy::too_many_arguments)]
fn emit_merged_flat_rect(
    out: &mut Vec<TriMeshVertex>,
    grid: &[[Option<CandidateQuad>; CHUNK_DIM]; CHUNK_DIM],
    u0: usize,
    v0: usize,
    w: usize,
    h: usize,
    axis: usize,
    positive: bool,
) {
    let Some(q00) = grid[v0][u0] else { return };
    let Some(q10) = grid[v0][u0 + w - 1] else { return };
    let Some(q11) = grid[v0 + h - 1][u0 + w - 1] else { return };
    let Some(q01) = grid[v0 + h - 1][u0] else { return };

    let mut n_sum = Vec3::ZERO;
    let mut pt_sum = Vec3::ZERO;
    let mut cnt = 0.0f32;
    for dv in 0..h {
        for du in 0..w {
            if let Some(q) = grid[v0 + dv][u0 + du] {
                n_sum += q.normal;
                pt_sum += (q.p0 + q.p1 + q.p2 + q.p3) * 0.25;
                cnt += 1.0;
            }
        }
    }
    let avg_pt = if cnt > 0.0 { pt_sum / cnt } else { (q00.p0 + q11.p2) * 0.5 };
    let fallback_n = match (axis, positive) {
        (0, true) => Vec3::X,
        (0, false) => Vec3::NEG_X,
        (1, true) => Vec3::Y,
        (1, false) => Vec3::NEG_Y,
        (_, true) => Vec3::Z,
        (_, false) => Vec3::NEG_Z,
    };
    let n = {
        let cand = n_sum.normalize_or_zero();
        if cand.length_squared() > 1e-4 { cand } else { fallback_n }
    };

    // Project the 4 outer corners cleanly onto the true fitted 3D plane (works for both straight & angled walls!)
    let snap = |p: Vec3| -> Vec3 { p - n * (p - avg_pt).dot(n) };
    // Match exact outward CCW winding from our verified (+X, +Y, +Z) quad corner layout:
    // +X (u=y, v=z): p0=(u0,v0).p0, p1=(u1,v0).p1, p2=(u1,v1).p2, p3=(u0,v1).p3
    // +Y (u=z, v=x): p0=(u0,v0).p0, p1=(u1,v0).p1, p2=(u1,v1).p2, p3=(u0,v1).p3
    // +Z (u=x, v=y): p0=(u0,v0).p0, p1=(u1,v0).p1, p2=(u1,v1).p2, p3=(u0,v1).p3
    let c0 = snap(q00.p0);
    let c1 = snap(q10.p1);
    let c2 = snap(q11.p2);
    let c3 = snap(q01.p3);

    let col = adaptive_region_color(true, false, 0.0, n);
    if positive {
        push_mesh_quad(out, c0, c1, c2, c3, n, col);
    } else {
        push_mesh_quad(out, c0, c3, c2, c1, n, col);
    }
}

/// Processes one `16×16` slice of candidate quads for `(axis, positive)`:
/// 1. **Flat / simple cells:** merged via 2D Greedy Maximal Rectangle Merging into minimal unified flat sheets (`flat_tris`).
/// 2. **Close-up bumpy cells (`min_dist_cm <= 95cm`):** emitted at full ultra-fine `2.5cm` resolution (`detail_tris`).
/// 3. **Farther bumpy cells (`min_dist_cm > 95cm`):** coalesced `2×2` into clean `5.0cm` quads (`detail_tris`).
fn emit_adaptive_slice(
    grid: &[[Option<CandidateQuad>; CHUNK_DIM]; CHUNK_DIM],
    axis: usize,
    positive: bool,
    flat_out: &mut Vec<TriMeshVertex>,
    detail_out: &mut Vec<TriMeshVertex>,
) {
    // Pass 1: Greedy coplanar rectangle merge for flat / simple regions
    let mut used = [[false; CHUNK_DIM]; CHUNK_DIM];
    for v in 0..CHUNK_DIM {
        let mut u = 0;
        while u < CHUNK_DIM {
            if used[v][u] {
                u += 1;
                continue;
            }
            let Some(q0) = grid[v][u] else {
                u += 1;
                continue;
            };
            if !q0.is_flat {
                u += 1;
                continue;
            }
            let bin = q0.plane_bin;
            let mut w = 1;
            while u + w < CHUNK_DIM && !used[v][u + w] {
                if let Some(q) = grid[v][u + w] {
                    if q.is_flat && (q.plane_bin - bin).abs() <= 1 && q.normal.dot(q0.normal) >= 0.94 {
                        w += 1;
                        continue;
                    }
                }
                break;
            }
            let mut h = 1;
            'grow_v: while v + h < CHUNK_DIM {
                for k in 0..w {
                    if used[v + h][u + k] {
                        break 'grow_v;
                    }
                    match grid[v + h][u + k] {
                        Some(q) if q.is_flat && (q.plane_bin - bin).abs() <= 1 && q.normal.dot(q0.normal) >= 0.94 => {}
                        _ => break 'grow_v,
                    }
                }
                h += 1;
            }
            for dv in 0..h {
                for du in 0..w {
                    used[v + dv][u + du] = true;
                }
            }
            emit_merged_flat_rect(flat_out, grid, u, v, w, h, axis, positive);
            u += w;
        }
    }

    // Pass 2: Distance-adaptive bumpy / curved cells:
    // - Farther (> 95cm) 2x2 blocks coalesce into 5cm quads
    // - Close-up (<= 95cm) cells emit full ultra-fine 2.5cm triangles!
    let mut detail_used = [[false; CHUNK_DIM]; CHUNK_DIM];
    for v0 in (0..CHUNK_DIM).step_by(2) {
        for u0 in (0..CHUNK_DIM).step_by(2) {
            if u0 + 1 < CHUNK_DIM && v0 + 1 < CHUNK_DIM {
                if let (Some(q00), Some(q10), Some(q11), Some(q01)) = (
                    grid[v0][u0],
                    grid[v0][u0 + 1],
                    grid[v0 + 1][u0 + 1],
                    grid[v0 + 1][u0],
                ) {
                    let any_flat = used[v0][u0] || used[v0][u0 + 1] || used[v0 + 1][u0 + 1] || used[v0 + 1][u0];
                    let min_d = q00.min_dist_cm.min(q10.min_dist_cm).min(q11.min_dist_cm).min(q01.min_dist_cm);
                    // Only coalesce 2x2 when ALL 4 cells are mid/far range (> 95cm) -> shrinks to 2.5cm as you move closer!
                    if !any_flat && min_d > CLOSE_DETAIL_DIST_CM {
                        detail_used[v0][u0] = true;
                        detail_used[v0][u0 + 1] = true;
                        detail_used[v0 + 1][u0 + 1] = true;
                        detail_used[v0 + 1][u0] = true;
                        let n = (q00.normal + q10.normal + q11.normal + q01.normal).normalize_or_zero();
                        let n = if n.length_squared() > 1e-4 { n } else { q00.normal };
                        let curv = (q00.curv + q10.curv + q11.curv + q01.curv) * 0.25;
                        let col = adaptive_region_color(false, false, curv, n);
                        if positive {
                            push_mesh_quad(detail_out, q00.p0, q10.p1, q11.p2, q01.p3, n, col);
                        } else {
                            push_mesh_quad(detail_out, q00.p0, q01.p3, q11.p2, q10.p1, n, col);
                        }
                    }
                }
            }
        }
    }

    for v in 0..CHUNK_DIM {
        for u in 0..CHUNK_DIM {
            if used[v][u] || detail_used[v][u] {
                continue;
            }
            if let Some(q) = grid[v][u] {
                let is_close = q.min_dist_cm <= CLOSE_DETAIL_DIST_CM;
                let col = adaptive_region_color(false, is_close, q.curv, q.normal);
                if positive {
                    push_mesh_quad(detail_out, q.p0, q.p1, q.p2, q.p3, q.normal, col);
                } else {
                    push_mesh_quad(detail_out, q.p0, q.p3, q.p2, q.p1, q.normal, col);
                }
            }
        }
    }
}

impl SpatialMeshMap {
    pub fn clear(&mut self) {
        self.planes.clear();
        self.chunks.clear();
        self.has_dense_depth = false;
        self.last_floor_y = None;
        self.mesh_gen = self.mesh_gen.wrapping_add(1);
    }

    pub fn clear_change_votes(&mut self) {
        for chunk in self.chunks.values_mut() {
            chunk.change_hits.fill(0);
        }
    }

    pub fn active_chunk_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|c| c.surface_count > 0 || c.total_tris_len() > 0)
            .count()
    }

    pub fn confirmed_voxel_count(&self) -> usize {
        self.chunks.values().map(|c| c.surface_count as usize).sum()
    }

    pub fn canonical_plane_count(&self) -> usize {
        self.planes.len()
    }

    pub fn inferred_floor_y(&self) -> Option<f32> {
        self.planes
            .iter()
            .filter(|p| p.kind == SurfaceKind::Floor)
            .map(|p| p.pos.y)
            .reduce(f32::min)
    }

    pub fn classify_surface_point(&self, _pos: Vec3, normal: Vec3, _cam_y: f32) -> SurfaceKind {
        let n = normal.normalize_or_zero();
        if n.length_squared() < 1e-4 {
            return SurfaceKind::Unknown;
        }
        if n.y >= 0.65 {
            SurfaceKind::Table
        } else if n.y <= -0.65 {
            SurfaceKind::Ceiling
        } else {
            SurfaceKind::VerticalObject
        }
    }

    /// Reads normalized TSDF value `[-1.0..+1.0]` for cell `(gx, gy, gz)` ONLY if that voxel is locked
    /// (`weight >= MESH_MIN_WEIGHT`) inside a chunk that actually has rendered triangles!
    #[inline]
    fn sample_locked_voxel_tsdf(&self, gx: i32, gy: i32, gz: i32) -> Option<f32> {
        let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
        let chunk = self.chunks.get(&ckey)?;
        if chunk.total_tris_len() == 0 {
            return None;
        }
        let idx = VoxelChunk::idx(lx, ly, lz);
        if chunk.weight[idx] < MESH_MIN_WEIGHT {
            return None;
        }
        Some(chunk.tsdf[idx] as f32 / 127.0)
    }

    /// Computes smooth outward unit normal of the scanned TSDF isosurface at world point `p`.
    fn sample_tsdf_gradient_normal(&self, p: Vec3) -> Vec3 {
        let (gx, gy, gz) = Self::world_to_cell(p);
        let center = self.sample_locked_voxel_tsdf(gx, gy, gz).unwrap_or(0.0);
        let dx = match (
            self.sample_locked_voxel_tsdf(gx + 1, gy, gz),
            self.sample_locked_voxel_tsdf(gx - 1, gy, gz),
        ) {
            (Some(a), Some(b)) => a - b,
            (Some(a), None) => (a - center) * 2.0,
            (None, Some(b)) => (center - b) * 2.0,
            (None, None) => 0.0,
        };
        let dy = match (
            self.sample_locked_voxel_tsdf(gx, gy + 1, gz),
            self.sample_locked_voxel_tsdf(gx, gy - 1, gz),
        ) {
            (Some(a), Some(b)) => a - b,
            (Some(a), None) => (a - center) * 2.0,
            (None, Some(b)) => (center - b) * 2.0,
            (None, None) => 0.0,
        };
        let dz = match (
            self.sample_locked_voxel_tsdf(gx, gy, gz + 1),
            self.sample_locked_voxel_tsdf(gx, gy, gz - 1),
        ) {
            (Some(a), Some(b)) => a - b,
            (Some(a), None) => (a - center) * 2.0,
            (None, Some(b)) => (center - b) * 2.0,
            (None, None) => 0.0,
        };
        Vec3::new(dx, dy, dz).normalize_or_zero()
    }

    /// Exact trilinear TSDF collision query for a rigid body sphere `(pos, radius)` against ONLY
    /// real scanned geometry (`total_tris_len() > 0 && weight >= MESH_MIN_WEIGHT`).
    /// Returns `Some((outward_normal, penetration))` when touching/penetrating a scanned surface,
    /// or `None` in unscanned empty air (ZERO fake/auto-generated planes!).
    pub fn sample_scanned_collision(&self, pos: Vec3, radius: f32) -> Option<(Vec3, f32)> {
        if !pos.is_finite() || self.chunks.is_empty() {
            return None;
        }
        let fx = pos.x / CELL_SIZE - 0.5;
        let fy = pos.y / CELL_SIZE - 0.5;
        let fz = pos.z / CELL_SIZE - 0.5;
        let gx0 = fx.floor() as i32;
        let gy0 = fy.floor() as i32;
        let gz0 = fz.floor() as i32;
        let tx = fx - gx0 as f32;
        let ty = fy - gy0 as f32;
        let tz = fz - gz0 as f32;

        let mut sdf_sum = 0.0f32;
        let mut w_sum = 0.0f32;
        let mut valid_cnt = 0u32;
        let mut grad = Vec3::ZERO;

        for ox in 0..=1i32 {
            let wx = if ox == 0 { 1.0 - tx } else { tx };
            let sx = if ox == 0 { -1.0 } else { 1.0 };
            for oy in 0..=1i32 {
                let wy = if oy == 0 { 1.0 - ty } else { ty };
                let sy = if oy == 0 { -1.0 } else { 1.0 };
                for oz in 0..=1i32 {
                    let wz = if oz == 0 { 1.0 - tz } else { tz };
                    let sz = if oz == 0 { -1.0 } else { 1.0 };
                    if let Some(s) = self.sample_locked_voxel_tsdf(gx0 + ox, gy0 + oy, gz0 + oz) {
                        let w = wx * wy * wz;
                        sdf_sum += s * w;
                        w_sum += w;
                        valid_cnt += 1;
                        grad += Vec3::new(sx * wy * wz, sy * wx * wz, sz * wx * wy) * s;
                    }
                }
            }
        }

        if valid_cnt < 4 || w_sum < 0.35 {
            return None;
        }
        let signed_dist = (sdf_sum / w_sum) * TSDF_TRUNC;
        if signed_dist >= radius || signed_dist <= -0.062 {
            return None;
        }

        let mut n = grad.normalize_or_zero();
        if n.length_squared() < 1e-4 {
            n = self.sample_tsdf_gradient_normal(pos);
        }
        if n.length_squared() < 1e-4 {
            return None;
        }

        // Floor is never rendered -> never acts as an invisible fake floor collider!
        if let Some(fy_floor) = self.inferred_floor_y() {
            if n.y >= 0.78 && (pos.y - radius) <= fy_floor + 0.052 {
                return None;
            }
        }

        // Confirm real rendered triangles exist at the surface contact point
        let contact = pos - n * signed_dist.max(0.0);
        let (cgx, cgy, cgz) = Self::world_to_cell(contact);
        let (ckey, _, _, _) = Self::split_cell(cgx, cgy, cgz);
        let has_mesh = self
            .chunks
            .get(&ckey)
            .map(|c| c.total_tris_len() > 0)
            .unwrap_or(false);
        if !has_mesh {
            return None;
        }

        Some((n, (radius - signed_dist).clamp(0.0, radius + 0.04)))
    }

    /// Raycasts directly through the ACTUAL SCANNED TSDF MESH (`total_tris_len() > 0`).
    /// Returns `Some((hit_pos, outward_normal, hit_distance))` ONLY where real scanned mesh exists;
    /// returns `None` in unscanned empty space (used by both the 120Hz aiming reticle and real contact shadows!).
    pub fn raycast_scanned_surface(
        &self,
        origin: Vec3,
        dir: Vec3,
        min_dist: f32,
        max_dist: f32,
    ) -> Option<(Vec3, Vec3, f32)> {
        if !origin.is_finite() || self.chunks.is_empty() || max_dist <= min_dist {
            return None;
        }
        let d = dir.normalize_or_zero();
        if d.length_squared() < 1e-4 {
            return None;
        }
        let floor_y = self.inferred_floor_y();
        let step = 0.018f32; // 1.8cm (< 2.5cm cell)
        let mut t = min_dist.max(0.02);
        let mut prev_valid = false;
        let mut prev_t = t;
        let mut prev_sdf = 1.0f32;
        let mut miss_ckey = (i32::MAX, i32::MAX, i32::MAX);

        while t <= max_dist {
            let p = origin + d * t;
            let (gx, gy, gz) = Self::world_to_cell(p);
            let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
            if ckey == miss_ckey {
                prev_valid = false;
                t += step;
                continue;
            }
            let Some(chunk) = self.chunks.get(&ckey) else {
                miss_ckey = ckey;
                prev_valid = false;
                t += step;
                continue;
            };
            if chunk.total_tris_len() == 0 {
                miss_ckey = ckey;
                prev_valid = false;
                t += step;
                continue;
            }

            let idx = VoxelChunk::idx(lx, ly, lz);
            if chunk.weight[idx] >= MESH_MIN_WEIGHT {
                let cur_sdf = chunk.tsdf[idx] as f32 / 127.0;
                let crossed = prev_valid && prev_sdf > 0.0 && cur_sdf <= 0.0;
                let grazed_entry = !prev_valid && (-0.38..=0.06).contains(&cur_sdf);
                if crossed || grazed_entry {
                    let hit_t = if crossed {
                        let denom = prev_sdf - cur_sdf;
                        let frac = if denom > 1e-4 {
                            (prev_sdf / denom).clamp(0.0, 1.0)
                        } else {
                            0.5
                        };
                        prev_t + (t - prev_t) * frac
                    } else {
                        t
                    };
                    let hit_pos = origin + d * hit_t;
                    let mut n = self.sample_tsdf_gradient_normal(hit_pos);
                    if n.length_squared() < 1e-4 {
                        n = -d;
                    }
                    // Ignore bare floor level (never rendered visually)
                    let on_suppressed_floor = floor_y
                        .map(|fy| n.y >= 0.78 && hit_pos.y <= fy + 0.052)
                        .unwrap_or(false);
                    if !on_suppressed_floor && n.dot(-d) > -0.25 {
                        return Some((hit_pos, n, hit_t));
                    }
                }
                prev_valid = true;
                prev_t = t;
                prev_sdf = cur_sdf;
            } else {
                prev_valid = false;
            }
            t += step;
        }
        None
    }

    fn push_or_evict_plane(&mut self, plane: CanonicalPlane) {
        if self.planes.len() >= MAX_CANONICAL_PLANES {
            if let Some((idx, _)) = self
                .planes
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.is_explicit_patch
                        .cmp(&b.is_explicit_patch)
                        .reverse()
                        .then_with(|| a.last_seen_gen.cmp(&b.last_seen_gen))
                })
            {
                self.planes.swap_remove(idx);
            }
        }
        self.planes.push(plane);
    }

    /// Ingests ARCore plane polygons strictly for invisible floor height (`physics_3d`) and reticle snapping.
    /// When a new floor elevation is learned, marks floor-level chunks dirty so floor quads disappear immediately!
    pub fn ingest_ar_plane_polygons(&mut self, incoming: &[RawArPlane], cam_y: f32) {
        self.tick_gen = self.tick_gen.wrapping_add(1);
        if incoming.is_empty() {
            return;
        }

        let mut summary: Vec<(i32, Vec3, f32, f32)> =
            Vec::with_capacity(self.planes.len() + incoming.len());
        for existing in &self.planes {
            if existing.raw_type == 0 {
                summary.push((0, existing.pos, existing.extent_x, existing.extent_z));
            }
        }
        for p in incoming {
            if p.pos.is_finite() {
                summary.push((p.raw_type, p.pos, p.extent_x, p.extent_z));
            }
        }
        let floor_y = infer_floor_y(&summary, cam_y);
        let mut changed = false;

        for raw in incoming {
            if !raw.pos.is_finite() {
                continue;
            }
            let rot = if raw.rot.length_squared() > 1e-6 {
                raw.rot.normalize()
            } else {
                Quat::IDENTITY
            };
            let normal = (rot * Vec3::Y).normalize_or_zero();
            if normal.length_squared() < 1e-4 {
                continue;
            }
            let raw_poly = if raw.polygon_xz.len() >= 3 {
                convex_hull_2d(&raw.polygon_xz)
            } else {
                rect_fallback_poly(raw.extent_x, raw.extent_z)
            };
            if raw_poly.len() < 3 {
                continue;
            }
            let (_, area, ex, ez) = poly_metrics(&raw_poly);
            let kind = classify_ar_plane(raw.raw_type, raw.pos.y, ex, ez, cam_y, floor_y);
            if kind == SurfaceKind::Unknown {
                continue;
            }
            let new_rad = 0.5 * (ex * ex + ez * ez).sqrt();
            let match_idx = self.planes.iter().position(|old| {
                old.normal.dot(normal) >= 0.90
                    && (raw.pos - old.pos).dot(old.normal).abs() <= 0.075
                    && (raw.pos - old.pos).length() <= old.bounding_radius() + new_rad + 0.30
            });
            if let Some(idx) = match_idx {
                self.merge_into_plane(idx, raw.raw_type, raw.pos, rot, &raw_poly, area, cam_y, floor_y);
                changed = true;
            } else {
                self.push_or_evict_plane(CanonicalPlane {
                    kind,
                    raw_type: raw.raw_type,
                    pos: raw.pos,
                    rot,
                    normal,
                    polygon_xz: raw_poly,
                    extent_x: ex,
                    extent_z: ez,
                    area,
                    weight: area.clamp(0.15, 6.0),
                    last_seen_gen: self.tick_gen,
                    is_explicit_patch: false,
                    carve_hits: 0,
                    detail_hits: 0,
                    demoted_to_object: false,
                });
                changed = true;
            }
        }

        if changed {
            self.coalesce_overlapping_planes(cam_y, floor_y);
            let next_fy = self.inferred_floor_y();
            let floor_updated = match (self.last_floor_y, next_fy) {
                (None, Some(_)) => true,
                (Some(a), Some(b)) => (a - b).abs() > 0.03,
                _ => false,
            };
            if floor_updated {
                self.last_floor_y = next_fy;
                if let Some(fy) = next_fy {
                    let span = CHUNK_DIM as f32 * CELL_SIZE;
                    for (&ckey, chunk) in &mut self.chunks {
                        let y_min = ckey.1 as f32 * span;
                        let y_max = y_min + span;
                        if fy >= y_min - 0.08 && fy <= y_max + 0.08 {
                            chunk.dirty = true;
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_into_plane(
        &mut self,
        idx: usize,
        raw_type: i32,
        new_pos: Vec3,
        new_rot: Quat,
        new_poly: &[[f32; 2]],
        new_area: f32,
        cam_y: f32,
        floor_y: Option<f32>,
    ) {
        let old = &mut self.planes[idx];
        let old_ax_x = old.rot * Vec3::X;
        let old_ax_z = old.rot * Vec3::Z;
        let new_ax_x = new_rot * Vec3::X;
        let new_ax_z = new_rot * Vec3::Z;

        let w_new = new_area.clamp(0.12, 4.0);
        let alpha = (w_new / (old.weight + w_new)).clamp(0.12, 0.45);
        let plane_anchor = old.pos + old.normal * ((new_pos - old.pos).dot(old.normal) * alpha);

        let mut combined_xz: Vec<[f32; 2]> = Vec::with_capacity(old.polygon_xz.len() + new_poly.len());
        for &[lx, lz] in &old.polygon_xz {
            let wp = old.pos + old_ax_x * lx + old_ax_z * lz;
            let rel = wp - plane_anchor;
            combined_xz.push([rel.dot(old_ax_x), rel.dot(old_ax_z)]);
        }
        for &[lx, lz] in new_poly {
            let wp = new_pos + new_ax_x * lx + new_ax_z * lz;
            let rel = wp - plane_anchor;
            combined_xz.push([rel.dot(old_ax_x), rel.dot(old_ax_z)]);
        }

        let hull = convex_hull_2d(&combined_xz);
        if hull.len() >= 3 {
            let (center_xz, area, ex, ez) = poly_metrics(&hull);
            let shifted_pos = plane_anchor + old_ax_x * center_xz[0] + old_ax_z * center_xz[1];
            let recentered: Vec<[f32; 2]> = hull
                .into_iter()
                .map(|[x, z]| [x - center_xz[0], z - center_xz[1]])
                .collect();
            old.pos = shifted_pos;
            old.polygon_xz = recentered;
            old.extent_x = ex.min(8.0);
            old.extent_z = ez.min(8.0);
            old.area = area;
            old.weight = (old.weight + w_new * 0.25).min(20.0);
            old.is_explicit_patch = false;
            old.last_seen_gen = self.tick_gen;
            old.kind = classify_ar_plane(raw_type, old.pos.y, old.extent_x, old.extent_z, cam_y, floor_y);
        }
    }

    fn coalesce_overlapping_planes(&mut self, cam_y: f32, floor_y: Option<f32>) {
        let mut i = 0;
        while i < self.planes.len() {
            let mut j = i + 1;
            while j < self.planes.len() {
                let a = &self.planes[i];
                let b = &self.planes[j];
                let same_plane = a.normal.dot(b.normal) > 0.92
                    && (b.pos - a.pos).dot(a.normal).abs() < 0.065
                    && (b.pos - a.pos).length() <= a.bounding_radius() + b.bounding_radius() + 0.15;
                if same_plane {
                    let b_removed = self.planes.swap_remove(j);
                    self.merge_into_plane(
                        i,
                        b_removed.raw_type,
                        b_removed.pos,
                        b_removed.rot,
                        &b_removed.polygon_xz,
                        b_removed.area,
                        cam_y,
                        floor_y,
                    );
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }

    #[inline]
    fn world_to_cell(p: Vec3) -> (i32, i32, i32) {
        (
            (p.x / CELL_SIZE).floor() as i32,
            (p.y / CELL_SIZE).floor() as i32,
            (p.z / CELL_SIZE).floor() as i32,
        )
    }

    #[inline]
    fn cell_center(gx: i32, gy: i32, gz: i32) -> Vec3 {
        Vec3::new(
            (gx as f32 + 0.5) * CELL_SIZE,
            (gy as f32 + 0.5) * CELL_SIZE,
            (gz as f32 + 0.5) * CELL_SIZE,
        )
    }

    #[inline]
    fn split_cell(gx: i32, gy: i32, gz: i32) -> ((i32, i32, i32), usize, usize, usize) {
        let d = CHUNK_DIM as i32;
        (
            (gx.div_euclid(d), gy.div_euclid(d), gz.div_euclid(d)),
            gx.rem_euclid(d) as usize,
            gy.rem_euclid(d) as usize,
            gz.rem_euclid(d) as usize,
        )
    }

    fn get_or_alloc_chunk(&mut self, ckey: (i32, i32, i32), gen: u64) -> Option<&mut VoxelChunk> {
        if !self.chunks.contains_key(&ckey) {
            if self.chunks.len() >= MAX_ACTIVE_CHUNKS {
                if let Some(&evict_key) = self
                    .chunks
                    .iter()
                    .min_by_key(|(_, c)| (c.surface_count > 0, c.last_touched_gen))
                    .map(|(k, _)| k)
                {
                    self.chunks.remove(&evict_key);
                }
            }
            self.chunks.insert(ckey, VoxelChunk::new(gen));
        }
        self.chunks.get_mut(&ckey)
    }

    fn mark_border_neighbors_dirty(&mut self, border_flips: &[((i32, i32, i32), usize, usize, usize)]) {
        const LAST: usize = CHUNK_DIM - 1;
        for &(ckey, lx, ly, lz) in border_flips {
            let dx_r = if lx == 0 { -1..=0 } else if lx == LAST { 0..=1 } else { 0..=0 };
            let dy_r = if ly == 0 { -1..=0 } else if ly == LAST { 0..=1 } else { 0..=0 };
            let dz_r = if lz == 0 { -1..=0 } else if lz == LAST { 0..=1 } else { 0..=0 };
            for dx in dx_r {
                for dy in dy_r.clone() {
                    for dz in dz_r.clone() {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if let Some(nb) = self.chunks.get_mut(&(ckey.0 + dx, ckey.1 + dy, ckey.2 + dz)) {
                            nb.dirty = true;
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn integrate_surface_band(
        &mut self,
        cam_pos: Vec3,
        dir: Vec3,
        d_obs: f32,
        w: f32,
        gen: u64,
        border_flips: &mut Vec<((i32, i32, i32), usize, usize, usize)>,
    ) -> u32 {
        let obs_dist_cm = (d_obs * 100.0).round().clamp(15.0, 254.0) as u8;
        let t_start = (d_obs - TSDF_TRUNC * 0.96).max(0.16);
        let t_end = d_obs + TSDF_TRUNC * 0.78;
        let step = CELL_SIZE * 0.80; // 2.0cm step (< 2.5cm cell)

        let mut prev_cell = (i32::MAX, i32::MAX, i32::MAX);
        let mut updated = 0u32;
        let mut t = t_start;
        while t <= t_end {
            let cell = Self::world_to_cell(cam_pos + dir * t);
            if cell != prev_cell {
                prev_cell = cell;
                let center = Self::cell_center(cell.0, cell.1, cell.2);
                let ray_t = (center - cam_pos).dot(dir);
                let sdf = ((d_obs - ray_t) / TSDF_TRUNC).clamp(-1.0, 1.0);
                let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
                if let Some(chunk) = self.get_or_alloc_chunk(ckey, gen) {
                    let idx = VoxelChunk::idx(lx, ly, lz);
                    if chunk.fuse_voxel(idx, sdf, w, obs_dist_cm, false, gen) {
                        updated += 1;
                        if lx == 0
                            || ly == 0
                            || lz == 0
                            || lx == CHUNK_DIM - 1
                            || ly == CHUNK_DIM - 1
                            || lz == CHUNK_DIM - 1
                        {
                            border_flips.push((ckey, lx, ly, lz));
                        }
                    }
                }
            }
            t += step;
        }
        updated
    }

    /// Zero-dead-zone free-space ray carving:
    /// - Turbo (`diff_only = false`): carves right up to `d_obs - 5.5cm` (overlapping seamlessly with
    ///   `integrate_surface_band`'s `+sdf` front band!) and hard-erodes floating errors in 1-2 frames!
    /// - Locked (`diff_only = true`): requires 3 consecutive ticks (`LOCK_CHANGE_CONFIRM`) before carving.
    #[allow(clippy::too_many_arguments)]
    fn carve_free_space_ray(
        &mut self,
        cam_pos: Vec3,
        dir: Vec3,
        d_obs: f32,
        carve_w: f32,
        diff_only: bool,
        gen: u64,
        border_flips: &mut Vec<((i32, i32, i32), usize, usize, usize)>,
    ) -> u32 {
        let margin = if diff_only { 0.12 } else { 0.055 };
        let max_t = (d_obs - margin).min(2.65);
        if max_t <= 0.18 {
            return 0;
        }
        let step = CELL_SIZE * 0.85; // 2.1cm step (< 2.5cm cell -> zero Swiss cheese!)
        let mut t = 0.18f32;
        let mut carved = 0u32;
        let mut steps = 0usize;
        let mut prev_cell = (i32::MAX, i32::MAX, i32::MAX);
        let mut miss_ckey = (i32::MAX, i32::MAX, i32::MAX);

        while t < max_t && steps < 118 {
            let cell = Self::world_to_cell(cam_pos + dir * t);
            if cell != prev_cell {
                prev_cell = cell;
                let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
                if ckey != miss_ckey {
                    if let Some(chunk) = self.chunks.get_mut(&ckey) {
                        let idx = VoxelChunk::idx(lx, ly, lz);
                        let old_w = chunk.weight[idx];
                        let old_t = chunk.tsdf[idx];
                        if diff_only {
                            if old_w >= MESH_MIN_WEIGHT && old_t <= 18 {
                                let votes = chunk.vote_change(idx, gen);
                                if votes >= LOCK_CHANGE_CONFIRM {
                                    if chunk.fuse_voxel(idx, 1.0, carve_w.max(36.0), 255, true, gen) {
                                        carved += 1;
                                        if lx == 0
                                            || ly == 0
                                            || lz == 0
                                            || lx == CHUNK_DIM - 1
                                            || ly == CHUNK_DIM - 1
                                            || lz == CHUNK_DIM - 1
                                        {
                                            border_flips.push((ckey, lx, ly, lz));
                                        }
                                    }
                                }
                            }
                        } else if old_w > 0 && old_t < 96 {
                            if chunk.fuse_voxel(idx, 1.0, carve_w.max(40.0), 255, true, gen) {
                                carved += 1;
                                if lx == 0
                                    || ly == 0
                                    || lz == 0
                                    || lx == CHUNK_DIM - 1
                                    || ly == CHUNK_DIM - 1
                                    || lz == CHUNK_DIM - 1
                                {
                                    border_flips.push((ckey, lx, ly, lz));
                                }
                            }
                        }
                    } else {
                        miss_ckey = ckey;
                    }
                }
            }
            t += step;
            steps += 1;
        }
        carved
    }

    fn match_locked_surface_along_ray(&mut self, cam_pos: Vec3, dir: Vec3, d_obs: f32) -> bool {
        let t_min = (d_obs - 0.10).max(0.18);
        let t_max = d_obs + 0.10;
        let step = CELL_SIZE * 0.90;
        let mut t = t_min;
        let mut prev_cell = (i32::MAX, i32::MAX, i32::MAX);
        let mut miss_ckey = (i32::MAX, i32::MAX, i32::MAX);

        while t <= t_max {
            let cell = Self::world_to_cell(cam_pos + dir * t);
            if cell != prev_cell {
                prev_cell = cell;
                let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
                if ckey != miss_ckey {
                    if let Some(chunk) = self.chunks.get_mut(&ckey) {
                        let idx = VoxelChunk::idx(lx, ly, lz);
                        if chunk.weight[idx] >= MESH_MIN_WEIGHT && (-105..=40).contains(&chunk.tsdf[idx]) {
                            chunk.change_hits[idx] = 0;
                            return true;
                        }
                    } else {
                        miss_ckey = ckey;
                    }
                }
            }
            t += step;
        }
        false
    }

    fn has_locked_surface_behind(&self, cam_pos: Vec3, dir: Vec3, d_obs: f32) -> bool {
        let t_start = d_obs + 0.12;
        let t_end = (d_obs + 1.80).min(3.10);
        let step = CELL_SIZE * 1.6;
        let mut t = t_start;
        let mut miss_ckey = (i32::MAX, i32::MAX, i32::MAX);

        while t <= t_end {
            let cell = Self::world_to_cell(cam_pos + dir * t);
            let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
            if ckey != miss_ckey {
                if let Some(chunk) = self.chunks.get(&ckey) {
                    let idx = VoxelChunk::idx(lx, ly, lz);
                    if chunk.weight[idx] >= MESH_MIN_WEIGHT && chunk.tsdf[idx] <= 24 {
                        return true;
                    }
                } else {
                    miss_ckey = ckey;
                }
            }
            t += step;
        }
        false
    }

    /// Primary Adaptive Multi-Resolution TSDF Scanner (`2.5cm` close-up <-> `5cm` mid-range <-> merged flat sheets, NO floor mesh!):
    /// - Moving closer overrides stale distant errors and unlocks ultra-fine `2.5cm` triangles!
    /// - Zero carving dead-zone (`d_obs - 5.5cm`) erases floating mistakes immediately when panning back!
    /// - Skips bare flat floor pixels (`hit_world.y <= floor_y + 4.5cm`) while still carving above the floor!
    #[allow(clippy::too_many_arguments)]
    pub fn ingest_depth_map(
        &mut self,
        depth_bytes: &[u8],
        dw: usize,
        dh: usize,
        d_row_stride: usize,
        d_pix_stride: usize,
        conf_bytes: Option<&[u8]>,
        cw: usize,
        ch: usize,
        c_row_stride: usize,
        c_pix_stride: usize,
        is_raw_only: bool,
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        img_w: usize,
        img_h: usize,
        cam_pos: Vec3,
        cam_rot: Quat,
        turbo: bool,
        stride: usize,
    ) -> DepthIngestStats {
        if dw < 8
            || dh < 8
            || img_w == 0
            || img_h == 0
            || fx.abs() < 1.0
            || fy.abs() < 1.0
            || !cam_pos.is_finite()
            || cam_rot.length_squared() < 1e-6
        {
            return DepthIngestStats::default();
        }

        self.tick_gen = self.tick_gen.wrapping_add(1);
        let gen = self.tick_gen;
        let rot = cam_rot.normalize();
        let floor_y = self.inferred_floor_y();

        let sx = dw as f32 / img_w as f32;
        let sy = dh as f32 / img_h as f32;
        let fx_d = fx * sx;
        let fy_d = fy * sy;
        let cx_d = cx * sx;
        let cy_d = cy * sy;

        let step_px = stride.clamp(1, 4);
        let pad_x = (dw / 24).max(2);
        let pad_y = (dh / 24).max(2);
        let max_depth_m = if turbo { 2.85f32 } else { 2.65f32 };

        let mut stats = DepthIngestStats::default();
        let mut border_flips: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::with_capacity(256);

        let mut y = pad_y;
        while y + pad_y < dh {
            let mut x = pad_x;
            while x + pad_x < dw {
                let d_mm = read_depth_mm(depth_bytes, x, y, d_row_stride, d_pix_stride);
                if (220..=2900).contains(&d_mm) {
                    let dm = d_mm as f32 * 0.001;
                    if dm <= max_depth_m {
                        let (conf_ok, base_w, allow_carve) = if let Some(cb) = conf_bytes {
                            let c0 = read_conf_u8(cb, x, y, dw, dh, cw, ch, c_row_stride, c_pix_stride);
                            if !turbo {
                                if c0 >= 65 {
                                    (true, 18.0f32, true)
                                } else if c0 >= 30
                                    && max_nearby_conf(cb, x, y, dw, dh, cw, ch, c_row_stride, c_pix_stride) >= 75
                                {
                                    (true, 14.0f32, true)
                                } else {
                                    (false, 0.0f32, false)
                                }
                            } else if is_raw_only {
                                if c0 >= 65 {
                                    (true, if c0 >= 130 { 18.0 } else { 13.0 }, true)
                                } else {
                                    (false, 0.0f32, false)
                                }
                            } else if c0 >= 90 {
                                (true, 18.0f32, true)
                            } else if c0 >= 35 {
                                (true, 13.0f32, true)
                            } else {
                                let nb_c = max_nearby_conf(cb, x, y, dw, dh, cw, ch, c_row_stride, c_pix_stride);
                                if nb_c >= 35 {
                                    (true, 10.0f32, nb_c >= 60)
                                } else {
                                    (false, 0.0f32, false)
                                }
                            }
                        } else {
                            (true, 11.0f32, true)
                        };

                        if !conf_ok {
                            x += step_px;
                            continue;
                        }

                        // ±1px silhouette jump guard: eliminates spiderwebs without clipping small objects!
                        let dl1 = read_depth_mm(depth_bytes, x - 1, y, d_row_stride, d_pix_stride);
                        let dr1 = read_depth_mm(depth_bytes, x + 1, y, d_row_stride, d_pix_stride);
                        let du1 = read_depth_mm(depth_bytes, x, y - 1, d_row_stride, d_pix_stride);
                        let dd1 = read_depth_mm(depth_bytes, x, y + 1, d_row_stride, d_pix_stride);
                        if dl1 == 0 || dr1 == 0 || du1 == 0 || dd1 == 0 {
                            x += step_px;
                            continue;
                        }
                        let max_jump = ((d_mm as f32) * 0.085).clamp(50.0, 155.0) as i32;
                        let edge_jump = (dl1 as i32 - d_mm as i32).abs() > max_jump
                            || (dr1 as i32 - d_mm as i32).abs() > max_jump
                            || (du1 as i32 - d_mm as i32).abs() > max_jump
                            || (dd1 as i32 - d_mm as i32).abs() > max_jump;

                        if !edge_jump {
                            let p_cam = Vec3::new(
                                (x as f32 - cx_d) / fx_d * dm,
                                (cy_d - y as f32) / fy_d * dm,
                                -dm,
                            );
                            let world_vec = rot * p_cam;
                            let dist = world_vec.length();
                            if dist >= 0.22 && dist.is_finite() {
                                let dir = world_vec / dist;
                                let hit_world = cam_pos + world_vec;
                                stats.valid_rays += 1;

                                // NO FLOOR MESH: skip integrating flat bare floor pixels (<= floor_y + 4.5cm),
                                // while still letting this ray carve any floating errors hovering above the floor!
                                let on_bare_floor = floor_y
                                    .map(|fy| hit_world.y <= fy + 0.045)
                                    .unwrap_or(false);

                                // Distance-proportional quality boost: close-up rays (<1.0m) carry much higher accuracy!
                                let dist_boost = if dist <= 0.95 { 1.25 } else { 1.0 };

                                if turbo {
                                    stats.changed_rays += 1;
                                    if allow_carve {
                                        stats.updated_voxels += self.carve_free_space_ray(
                                            cam_pos,
                                            dir,
                                            dist,
                                            40.0,
                                            false,
                                            gen,
                                            &mut border_flips,
                                        );
                                    }
                                    if !on_bare_floor {
                                        stats.updated_voxels += self.integrate_surface_band(
                                            cam_pos,
                                            dir,
                                            dist,
                                            base_w * dist_boost,
                                            gen,
                                            &mut border_flips,
                                        );
                                    }
                                } else {
                                    // TRUE LOCK MODE:
                                    // 1. Unchanged locked surface (±10cm) or bare floor -> 0 writes!
                                    if on_bare_floor || self.match_locked_surface_along_ray(cam_pos, dir, dist) {
                                        if on_bare_floor && allow_carve {
                                            let carved = self.carve_free_space_ray(
                                                cam_pos,
                                                dir,
                                                dist,
                                                36.0,
                                                true,
                                                gen,
                                                &mut border_flips,
                                            );
                                            if carved > 0 {
                                                stats.changed_rays += 1;
                                                stats.updated_voxels += carved;
                                            }
                                        }
                                        x += step_px;
                                        continue;
                                    }
                                    // 2. Confirmed 3-tick removal of old object in front of `dist`
                                    if allow_carve {
                                        let carved = self.carve_free_space_ray(
                                            cam_pos,
                                            dir,
                                            dist,
                                            36.0,
                                            true,
                                            gen,
                                            &mut border_flips,
                                        );
                                        if carved > 0 {
                                            stats.changed_rays += 1;
                                            stats.updated_voxels += carved;
                                        }
                                    }
                                    // 3. Confirmed 3-tick placement of new object in front of existing locked surface
                                    if self.has_locked_surface_behind(cam_pos, dir, dist) {
                                        let (gx, gy, gz) = Self::world_to_cell(hit_world);
                                        let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
                                        let votes = if let Some(chunk) = self.get_or_alloc_chunk(ckey, gen) {
                                            let idx = VoxelChunk::idx(lx, ly, lz);
                                            let v = chunk.vote_change(idx, gen);
                                            chunk.last_touched_gen = gen;
                                            v
                                        } else {
                                            0
                                        };
                                        if votes >= LOCK_CHANGE_CONFIRM {
                                            stats.changed_rays += 1;
                                            stats.updated_voxels += self.integrate_surface_band(
                                                cam_pos,
                                                dir,
                                                dist,
                                                22.0,
                                                gen,
                                                &mut border_flips,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                x += step_px;
            }
            y += step_px;
        }

        if stats.valid_rays >= 24 {
            self.has_dense_depth = true;
        }

        if !border_flips.is_empty() {
            self.mark_border_neighbors_dirty(&border_flips);
        }

        self.chunks.retain(|_, c| {
            c.surface_count > 0
                || c.total_tris_len() > 0
                || gen.saturating_sub(c.last_touched_gen) <= 45
        });

        stats
    }

    /// Fallback sparse SLAM PointCloud TSDF ingestion (only before Dense Depth warms up or on non-Depth devices).
    pub fn ingest_point_cloud(&mut self, point_count: usize, packed_xyzc: &[f32], cam_pos: Vec3) {
        if self.has_dense_depth {
            return;
        }
        self.tick_gen = self.tick_gen.wrapping_add(1);
        let gen = self.tick_gen;
        let floor_y = self.inferred_floor_y();
        let n = point_count.min(packed_xyzc.len() / 4);
        let mut border_flips: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::new();

        for i in 0..n {
            let base = i * 4;
            let conf = packed_xyzc[base + 3];
            if !conf.is_finite() || conf < 0.35 {
                continue;
            }
            let pt = Vec3::new(packed_xyzc[base], packed_xyzc[base + 1], packed_xyzc[base + 2]);
            if !pt.is_finite() {
                continue;
            }
            if let Some(fy) = floor_y {
                if pt.y <= fy + 0.045 {
                    continue;
                }
            }
            let delta = pt - cam_pos;
            let dist_sq = delta.length_squared();
            if !(0.22 * 0.22..=3.2 * 3.2).contains(&dist_sq) {
                continue;
            }
            let dist = dist_sq.sqrt();
            let dir = delta / dist;
            if i % 3 == 0 {
                self.carve_free_space_ray(cam_pos, dir, dist, 18.0, false, gen, &mut border_flips);
            }
            let w = if conf >= 0.72 { 16.0 } else { 12.0 };
            self.integrate_surface_band(cam_pos, dir, dist, w, gen, &mut border_flips);
        }

        if !border_flips.is_empty() {
            self.mark_border_neighbors_dirty(&border_flips);
        }
        self.chunks.retain(|_, c| {
            c.surface_count > 0 || c.total_tris_len() > 0 || gen.saturating_sub(c.last_touched_gen) <= 60
        });
    }

    pub fn step_greedy_meshing(&mut self, max_dirty_chunks: usize) -> usize {
        let dirty_keys: Vec<(i32, i32, i32)> = self
            .chunks
            .iter()
            .filter(|(_, c)| c.dirty)
            .take(max_dirty_chunks.max(1))
            .map(|(&k, _)| k)
            .collect();

        if dirty_keys.is_empty() {
            return 0;
        }

        let floor_y = self.inferred_floor_y();
        for ckey in &dirty_keys {
            let (detail_tris, flat_tris, aabbs) = self.build_chunk_adaptive_mesh(*ckey, floor_y);
            if let Some(chunk) = self.chunks.get_mut(ckey) {
                chunk.detail_tris = detail_tris;
                chunk.flat_tris = flat_tris;
                chunk.aabbs = aabbs;
                chunk.dirty = false;
            }
        }

        self.mesh_gen = self.mesh_gen.wrapping_add(1);
        dirty_keys.len()
    }

    /// Adaptive Multi-Resolution TSDF Surface Nets + Coplanar Flat Greedy Merger + Floating Island Pruner:
    /// 1. Prunes isolated floating mid-air islands (`< 7` voxels inside chunk).
    /// 2. Suppresses bare floor quads (`n.y >= 0.80 && y <= floor_y + 5cm`).
    /// 3. Merges flat / simple faces via 2D Greedy Rectangle Merging into 1 crisp unified sheet (`flat_tris`).
    /// 4. Keeps bumpy/curved regions at `2.5cm` micro-triangles when scanned up close (`<= 0.95m`), or `5.0cm`
    ///    coalesced quads when scanned only from afar (`detail_tris`)!
    fn build_chunk_adaptive_mesh(
        &self,
        ckey: (i32, i32, i32),
        floor_y: Option<f32>,
    ) -> (Vec<TriMeshVertex>, Vec<TriMeshVertex>, Vec<AabbCollider>) {
        let Some(chunk) = self.chunks.get(&ckey) else {
            return (Vec::new(), Vec::new(), Vec::new());
        };
        if chunk.surface_count == 0 {
            return (Vec::new(), Vec::new(), Vec::new());
        }

        let keep_mask = compute_non_floater_mask(chunk);

        let base_gx = ckey.0 * (CHUNK_DIM as i32);
        let base_gy = ckey.1 * (CHUNK_DIM as i32);
        let base_gz = ckey.2 * (CHUNK_DIM as i32);

        const H: usize = CHUNK_DIM + 2; // 18
        const H_VOL: usize = H * H * H;
        let h_idx = |x: usize, y: usize, z: usize| -> usize { (y * H + z) * H + x };

        let mut h_tsdf = [127i8; H_VOL];
        let mut h_weight = [0u8; H_VOL];
        let mut h_dist = [255u8; H_VOL];

        for ly in 0..CHUNK_DIM {
            for lz in 0..CHUNK_DIM {
                let src_row = (ly * CHUNK_DIM + lz) * CHUNK_DIM;
                let dst_row = ((ly + 1) * H + (lz + 1)) * H + 1;
                for lx in 0..CHUNK_DIM {
                    let si = src_row + lx;
                    // Pruned floating island specks are treated as empty air during meshing!
                    if VoxelChunk::is_near_surface(chunk.tsdf[si], chunk.weight[si]) && !keep_mask[si] {
                        h_tsdf[dst_row + lx] = 127;
                        h_weight[dst_row + lx] = 0;
                        h_dist[dst_row + lx] = 255;
                    } else {
                        h_tsdf[dst_row + lx] = chunk.tsdf[si];
                        h_weight[dst_row + lx] = chunk.weight[si];
                        h_dist[dst_row + lx] = chunk.min_dist_cm[si];
                    }
                }
            }
        }

        for dy in -1i32..=1 {
            for dz in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let Some(nb) = self.chunks.get(&(ckey.0 + dx, ckey.1 + dy, ckey.2 + dz)) else {
                        continue;
                    };
                    let (hx0, nx0, sx_len) = match dx {
                        -1 => (0usize, CHUNK_DIM - 1, 1usize),
                        0 => (1usize, 0usize, CHUNK_DIM),
                        _ => (CHUNK_DIM + 1, 0usize, 1usize),
                    };
                    let (hy0, ny0, sy_len) = match dy {
                        -1 => (0usize, CHUNK_DIM - 1, 1usize),
                        0 => (1usize, 0usize, CHUNK_DIM),
                        _ => (CHUNK_DIM + 1, 0usize, 1usize),
                    };
                    let (hz0, nz0, sz_len) = match dz {
                        -1 => (0usize, CHUNK_DIM - 1, 1usize),
                        0 => (1usize, 0usize, CHUNK_DIM),
                        _ => (CHUNK_DIM + 1, 0usize, 1usize),
                    };
                    for oy in 0..sy_len {
                        for oz in 0..sz_len {
                            let src = ((ny0 + oy) * CHUNK_DIM + (nz0 + oz)) * CHUNK_DIM + nx0;
                            let dst = ((hy0 + oy) * H + (hz0 + oz)) * H + hx0;
                            h_tsdf[dst..dst + sx_len].copy_from_slice(&nb.tsdf[src..src + sx_len]);
                            h_weight[dst..dst + sx_len].copy_from_slice(&nb.weight[src..src + sx_len]);
                            h_dist[dst..dst + sx_len].copy_from_slice(&nb.min_dist_cm[src..src + sx_len]);
                        }
                    }
                }
            }
        }

        // Step A: Dual vertices on 17x17x17 grid at exact linear tsdf = 0 zero-crossings
        const D: usize = CHUNK_DIM + 1;
        const D_VOL: usize = D * D * D;
        let d_idx = |x: usize, y: usize, z: usize| -> usize { (y * D + z) * D + x };

        let mut dual_pos: Vec<Option<Vec3>> = vec![None; D_VOL];
        let mut dual_norm: Vec<Vec3> = vec![Vec3::Y; D_VOL];
        let mut dual_dist: Vec<u8> = vec![255u8; D_VOL];

        const CUBE_EDGES: [(usize, usize); 12] = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (0, 2), (1, 3), (4, 6), (5, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ];
        const CORNER_OFF: [(usize, usize, usize); 8] = [
            (0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1),
            (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1),
        ];

        for ly in 0..D {
            for lz in 0..D {
                for lx in 0..D {
                    let mut c_tsdf = [0i8; 8];
                    let mut c_w = [0u8; 8];
                    let mut c_pos = [Vec3::ZERO; 8];
                    let mut min_d = 255u8;
                    let mut any_pos = false;
                    let mut any_neg = false;

                    for k in 0..8 {
                        let (ox, oy, oz) = CORNER_OFF[k];
                        let hi = h_idx(lx + ox, ly + oy, lz + oz);
                        let t = h_tsdf[hi];
                        let w = h_weight[hi];
                        c_tsdf[k] = t;
                        c_w[k] = w;
                        if w >= MESH_MIN_WEIGHT {
                            min_d = min_d.min(h_dist[hi]);
                            if t > 0 { any_pos = true; } else { any_neg = true; }
                        }
                        let gx = base_gx + (lx + ox) as i32 - 1;
                        let gy = base_gy + (ly + oy) as i32 - 1;
                        let gz = base_gz + (lz + oz) as i32 - 1;
                        c_pos[k] = Self::cell_center(gx, gy, gz);
                    }
                    if !(any_pos && any_neg) {
                        continue;
                    }

                    let mut cross_sum = Vec3::ZERO;
                    let mut cross_n = 0u32;
                    for &(a, b) in &CUBE_EDGES {
                        if c_w[a] < MESH_MIN_WEIGHT || c_w[b] < MESH_MIN_WEIGHT {
                            continue;
                        }
                        let ta = c_tsdf[a] as f32;
                        let tb = c_tsdf[b] as f32;
                        if (ta > 0.0) != (tb > 0.0) {
                            let denom = ta - tb;
                            if denom.abs() > 1e-3 {
                                let frac = (ta / denom).clamp(0.0, 1.0);
                                cross_sum += c_pos[a].lerp(c_pos[b], frac);
                                cross_n += 1;
                            }
                        }
                    }

                    if cross_n > 0 {
                        let di = d_idx(lx, ly, lz);
                        dual_pos[di] = Some(cross_sum / (cross_n as f32));
                        dual_dist[di] = min_d;

                        let mut grad = Vec3::ZERO;
                        for k in 0..8 {
                            if c_w[k] >= MESH_MIN_WEIGHT {
                                let (ox, oy, oz) = CORNER_OFF[k];
                                let sx = if ox == 0 { -1.0 } else { 1.0 };
                                let sy = if oy == 0 { -1.0 } else { 1.0 };
                                let sz = if oz == 0 { -1.0 } else { 1.0 };
                                grad += Vec3::new(sx, sy, sz) * (c_tsdf[k] as f32);
                            }
                        }
                        let n = grad.normalize_or_zero();
                        dual_norm[di] = if n.length_squared() > 1e-4 { n } else { Vec3::Y };
                    }
                }
            }
        }

        // Step B: Gentle 18% Laplacian polish + 3D normal curvature estimation
        let mut smooth_pos = dual_pos.clone();
        let mut dual_curv = vec![0.0f32; D_VOL];
        for ly in 1..(D - 1) {
            for lz in 1..(D - 1) {
                for lx in 1..(D - 1) {
                    let di = d_idx(lx, ly, lz);
                    let Some(cur) = dual_pos[di] else { continue };
                    let n_cur = dual_norm[di];
                    let mut nb_sum = Vec3::ZERO;
                    let mut curv_sum = 0.0f32;
                    let mut nb_cnt = 0u32;
                    for (ox, oy, oz) in [
                        (-1isize, 0, 0), (1, 0, 0),
                        (0, -1, 0), (0, 1, 0),
                        (0, 0, -1), (0, 0, 1),
                    ] {
                        let ni = d_idx(
                            (lx as isize + ox) as usize,
                            (ly as isize + oy) as usize,
                            (lz as isize + oz) as usize,
                        );
                        if let Some(np) = dual_pos[ni] {
                            nb_sum += np;
                            curv_sum += (1.0 - n_cur.dot(dual_norm[ni])).clamp(0.0, 1.0);
                            nb_cnt += 1;
                        }
                    }
                    if nb_cnt >= 2 {
                        smooth_pos[di] = Some(cur * 0.82 + (nb_sum / nb_cnt as f32) * 0.18);
                        dual_curv[di] = curv_sum / (nb_cnt as f32);
                    }
                }
            }
        }

        // Step C: Collect axis-aligned slices (+X/-X, +Y/-Y, +Z/-Z), drop floor quads, and run
        // Coplanar Greedy Merge (flat faces -> 1 unified quad) + Distance LOD (<=95cm: 2.5cm, >95cm: 5cm)!
        let mut detail_tris = Vec::with_capacity((chunk.surface_count as usize) * 3);
        let mut flat_tris = Vec::with_capacity(128);

        let is_floor_quad = |mid: Vec3, n: Vec3| -> bool {
            if let Some(fy) = floor_y {
                n.y >= 0.78 && mid.y <= fy + 0.052
            } else {
                false
            }
        };

        // Axis 0 (+X / -X slices indexed by lx, grid[lz][ly])
        for lx in 0..CHUNK_DIM {
            let mut pos_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut neg_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut any_pos = false;
            let mut any_neg = false;
            for lz in 0..CHUNK_DIM {
                for ly in 0..CHUNK_DIM {
                    let hi0 = h_idx(lx + 1, ly + 1, lz + 1);
                    let hix = h_idx(lx + 2, ly + 1, lz + 1);
                    if h_weight[hi0] < MESH_MIN_WEIGHT || h_weight[hix] < MESH_MIN_WEIGHT {
                        continue;
                    }
                    let s0 = h_tsdf[hi0] <= 0;
                    if s0 == (h_tsdf[hix] <= 0) {
                        continue;
                    }
                    let i0 = d_idx(lx + 1, ly, lz);
                    let i1 = d_idx(lx + 1, ly + 1, lz);
                    let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                    let i3 = d_idx(lx + 1, ly, lz + 1);
                    if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                        (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                    {
                        let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3]).normalize_or_zero();
                        let n = if n.length_squared() > 1e-4 { n } else if s0 { Vec3::X } else { Vec3::NEG_X };
                        let mid = (p0 + p2) * 0.5;
                        if is_floor_quad(mid, n) {
                            continue;
                        }
                        let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                        let min_d = dual_dist[i0].min(dual_dist[i1]).min(dual_dist[i2]).min(dual_dist[i3]);
                        let axis_dir = if s0 { Vec3::X } else { Vec3::NEG_X };
                        let is_flat = curv < 0.085 && n.dot(axis_dir) >= 0.74;
                        let q = CandidateQuad {
                            p0, p1, p2, p3, normal: n, curv, min_dist_cm: min_d, is_flat,
                            plane_bin: (mid.x / 0.015).round() as i16,
                        };
                        if s0 {
                            pos_grid[lz][ly] = Some(q);
                            any_pos = true;
                        } else {
                            neg_grid[lz][ly] = Some(q);
                            any_neg = true;
                        }
                    }
                }
            }
            if any_pos { emit_adaptive_slice(&pos_grid, 0, true, &mut flat_tris, &mut detail_tris); }
            if any_neg { emit_adaptive_slice(&neg_grid, 0, false, &mut flat_tris, &mut detail_tris); }
        }

        // Axis 1 (+Y / -Y slices indexed by ly, grid[lx][lz])
        for ly in 0..CHUNK_DIM {
            let mut pos_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut neg_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut any_pos = false;
            let mut any_neg = false;
            for lx in 0..CHUNK_DIM {
                for lz in 0..CHUNK_DIM {
                    let hi0 = h_idx(lx + 1, ly + 1, lz + 1);
                    let hiy = h_idx(lx + 1, ly + 2, lz + 1);
                    if h_weight[hi0] < MESH_MIN_WEIGHT || h_weight[hiy] < MESH_MIN_WEIGHT {
                        continue;
                    }
                    let s0 = h_tsdf[hi0] <= 0;
                    if s0 == (h_tsdf[hiy] <= 0) {
                        continue;
                    }
                    let i0 = d_idx(lx, ly + 1, lz);
                    let i1 = d_idx(lx, ly + 1, lz + 1);
                    let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                    let i3 = d_idx(lx + 1, ly + 1, lz);
                    if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                        (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                    {
                        let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3]).normalize_or_zero();
                        let n = if n.length_squared() > 1e-4 { n } else if s0 { Vec3::Y } else { Vec3::NEG_Y };
                        let mid = (p0 + p2) * 0.5;
                        if is_floor_quad(mid, n) {
                            continue;
                        }
                        let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                        let min_d = dual_dist[i0].min(dual_dist[i1]).min(dual_dist[i2]).min(dual_dist[i3]);
                        let axis_dir = if s0 { Vec3::Y } else { Vec3::NEG_Y };
                        let is_flat = curv < 0.085 && n.dot(axis_dir) >= 0.74;
                        let q = CandidateQuad {
                            p0, p1, p2, p3, normal: n, curv, min_dist_cm: min_d, is_flat,
                            plane_bin: (mid.y / 0.015).round() as i16,
                        };
                        if s0 {
                            pos_grid[lx][lz] = Some(q);
                            any_pos = true;
                        } else {
                            neg_grid[lx][lz] = Some(q);
                            any_neg = true;
                        }
                    }
                }
            }
            if any_pos { emit_adaptive_slice(&pos_grid, 1, true, &mut flat_tris, &mut detail_tris); }
            if any_neg { emit_adaptive_slice(&neg_grid, 1, false, &mut flat_tris, &mut detail_tris); }
        }

        // Axis 2 (+Z / -Z slices indexed by lz, grid[ly][lx])
        for lz in 0..CHUNK_DIM {
            let mut pos_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut neg_grid = [[None; CHUNK_DIM]; CHUNK_DIM];
            let mut any_pos = false;
            let mut any_neg = false;
            for ly in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    let hi0 = h_idx(lx + 1, ly + 1, lz + 1);
                    let hiz = h_idx(lx + 1, ly + 1, lz + 2);
                    if h_weight[hi0] < MESH_MIN_WEIGHT || h_weight[hiz] < MESH_MIN_WEIGHT {
                        continue;
                    }
                    let s0 = h_tsdf[hi0] <= 0;
                    if s0 == (h_tsdf[hiz] <= 0) {
                        continue;
                    }
                    let i0 = d_idx(lx, ly, lz + 1);
                    let i1 = d_idx(lx + 1, ly, lz + 1);
                    let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                    let i3 = d_idx(lx, ly + 1, lz + 1);
                    if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                        (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                    {
                        let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3]).normalize_or_zero();
                        let n = if n.length_squared() > 1e-4 { n } else if s0 { Vec3::Z } else { Vec3::NEG_Z };
                        let mid = (p0 + p2) * 0.5;
                        if is_floor_quad(mid, n) {
                            continue;
                        }
                        let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                        let min_d = dual_dist[i0].min(dual_dist[i1]).min(dual_dist[i2]).min(dual_dist[i3]);
                        let axis_dir = if s0 { Vec3::Z } else { Vec3::NEG_Z };
                        let is_flat = curv < 0.085 && n.dot(axis_dir) >= 0.74;
                        let q = CandidateQuad {
                            p0, p1, p2, p3, normal: n, curv, min_dist_cm: min_d, is_flat,
                            plane_bin: (mid.z / 0.015).round() as i16,
                        };
                        if s0 {
                            pos_grid[ly][lx] = Some(q);
                            any_pos = true;
                        } else {
                            neg_grid[ly][lx] = Some(q);
                            any_neg = true;
                        }
                    }
                }
            }
            if any_pos { emit_adaptive_slice(&pos_grid, 2, true, &mut flat_tris, &mut detail_tris); }
            if any_neg { emit_adaptive_slice(&neg_grid, 2, false, &mut flat_tris, &mut detail_tris); }
        }

        // Step D: Merged per-layer XZ physics AABBs (only from non-floater surface voxels!)
        let mut aabbs = Vec::with_capacity(16);
        for ly in 0..CHUNK_DIM {
            let mut mask = [[false; CHUNK_DIM]; CHUNK_DIM];
            let mut any = false;
            for lz in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    let idx = VoxelChunk::idx(lx, ly, lz);
                    if keep_mask[idx]
                        && chunk.weight[idx] >= MESH_MIN_WEIGHT
                        && (-75..=14).contains(&chunk.tsdf[idx])
                    {
                        mask[lz][lx] = true;
                        any = true;
                    }
                }
            }
            if !any {
                continue;
            }
            for lz in 0..CHUNK_DIM {
                let mut lx = 0;
                while lx < CHUNK_DIM {
                    if !mask[lz][lx] {
                        lx += 1;
                        continue;
                    }
                    let mut w = 1;
                    while lx + w < CHUNK_DIM && mask[lz][lx + w] {
                        w += 1;
                    }
                    let mut h = 1;
                    'grow_z: while lz + h < CHUNK_DIM {
                        for k in 0..w {
                            if !mask[lz + h][lx + k] {
                                break 'grow_z;
                            }
                        }
                        h += 1;
                    }
                    for dz in 0..h {
                        for dx in 0..w {
                            mask[lz + dz][lx + dx] = false;
                        }
                    }
                    let min = Vec3::new(
                        (base_gx + lx as i32) as f32 * CELL_SIZE,
                        (base_gy + ly as i32) as f32 * CELL_SIZE,
                        (base_gz + lz as i32) as f32 * CELL_SIZE,
                    );
                    let max = min + Vec3::new(w as f32 * CELL_SIZE, CELL_SIZE, h as f32 * CELL_SIZE);
                    aabbs.push(AabbCollider {
                        min,
                        max,
                        is_object_box: false,
                    });
                    lx += w;
                }
            }
        }

        (detail_tris, flat_tris, aabbs)
    }

    pub fn snapshot_plane_markers(&self, _include_all_types: bool) -> Vec<PlaneMarker> {
        self.planes
            .iter()
            .filter(|p| {
                matches!(p.kind, SurfaceKind::Floor | SurfaceKind::Table | SurfaceKind::Seat | SurfaceKind::Wall)
                    || p.is_explicit_patch
            })
            .map(CanonicalPlane::to_plane_marker)
            .collect()
    }

    pub fn collect_physics_chunk_aabbs(&self, cam_pos: Vec3) -> Vec<AabbCollider> {
        let chunk_span = CHUNK_DIM as f32 * CELL_SIZE;
        let mut nearby: Vec<(f32, &VoxelChunk)> = self
            .chunks
            .iter()
            .filter_map(|(&ckey, chunk)| {
                if chunk.aabbs.is_empty() {
                    return None;
                }
                let center = Vec3::new(
                    (ckey.0 as f32 + 0.5) * chunk_span,
                    (ckey.1 as f32 + 0.5) * chunk_span,
                    (ckey.2 as f32 + 0.5) * chunk_span,
                );
                let d2 = (center - cam_pos).length_squared();
                if d2 <= 3.0 * 3.0 { Some((d2, chunk)) } else { None }
            })
            .collect();
        nearby.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut out = Vec::with_capacity(96);
        for (_, chunk) in nearby {
            for &b in &chunk.aabbs {
                if out.len() >= 128 {
                    return out;
                }
                out.push(b);
            }
        }
        out
    }

    /// Builds nearest-first Adaptive Mesh VBO returning `(active_chunks, detail_tris, flat_merged_tris)`.
    /// ZERO floor sheets and ZERO stuck ARCore plane rectangles!
    pub fn build_triangle_mesh_buffer(
        &self,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        out: &mut Vec<TriMeshVertex>,
    ) -> (u32, u32, u32) {
        out.clear();

        let chunk_span = CHUNK_DIM as f32 * CELL_SIZE;
        let mut visible_chunks: Vec<(f32, &VoxelChunk)> = Vec::with_capacity(self.chunks.len());
        for (&ckey, chunk) in &self.chunks {
            if chunk.total_tris_len() == 0 {
                continue;
            }
            let center = Vec3::new(
                (ckey.0 as f32 + 0.5) * chunk_span,
                (ckey.1 as f32 + 0.5) * chunk_span,
                (ckey.2 as f32 + 0.5) * chunk_span,
            );
            let to_c = center - cam_pos;
            let dist = to_c.length();
            if dist > MAX_RENDER_RADIUS_M {
                continue;
            }
            if dist > 1.0 && to_c.dot(cam_fwd) < -0.9 {
                continue;
            }
            visible_chunks.push((dist, chunk));
        }

        visible_chunks.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut active_chunks = 0u32;
        let mut detail_verts = 0usize;
        let mut flat_verts = 0usize;

        for (_, chunk) in visible_chunks {
            active_chunks += 1;
            let remain_d = MAX_MESH_VERTICES.saturating_sub(out.len());
            let copy_d = (chunk.detail_tris.len().min(remain_d) / 3) * 3;
            if copy_d > 0 {
                out.extend_from_slice(&chunk.detail_tris[..copy_d]);
                detail_verts += copy_d;
            }
            let remain_f = MAX_MESH_VERTICES.saturating_sub(out.len());
            let copy_f = (chunk.flat_tris.len().min(remain_f) / 3) * 3;
            if copy_f > 0 {
                out.extend_from_slice(&chunk.flat_tris[..copy_f]);
                flat_verts += copy_f;
            }
            if out.len() + 3 > MAX_MESH_VERTICES {
                break;
            }
        }

        (active_chunks, (detail_verts / 3) as u32, (flat_verts / 3) as u32)
    }

    pub fn export_tsdf_chunks(&self) -> Vec<((i32, i32, i32), Vec<(u16, i8, u8)>)> {
        let mut out = Vec::new();
        for (&ckey, chunk) in &self.chunks {
            if chunk.surface_count == 0 && chunk.total_tris_len() == 0 {
                continue;
            }
            let mut entries = Vec::with_capacity((chunk.surface_count as usize) * 2);
            for idx in 0..CHUNK_VOLUME {
                let w = chunk.weight[idx];
                let t = chunk.tsdf[idx];
                if w >= MESH_MIN_WEIGHT && t.abs() <= 118 {
                    entries.push((idx as u16, t, w));
                }
            }
            if !entries.is_empty() {
                out.push((ckey, entries));
            }
        }
        out
    }

    pub fn import_and_merge_transformed(
        &mut self,
        saved_planes: &[CanonicalPlane],
        saved_voxels: &[SavedTsdfVoxel],
        q_align: Quat,
        t_align: Vec3,
        cam_y: f32,
    ) {
        let raw_planes: Vec<RawArPlane> = saved_planes
            .iter()
            .map(|p| RawArPlane {
                raw_type: p.raw_type,
                pos: q_align * p.pos + t_align,
                rot: (q_align * p.rot).normalize(),
                extent_x: p.extent_x,
                extent_z: p.extent_z,
                polygon_xz: p.polygon_xz.clone(),
            })
            .collect();
        self.ingest_ar_plane_polygons(&raw_planes, cam_y);

        let gen = self.tick_gen;
        let mut imported_voxels = 0usize;
        for &sv in saved_voxels {
            let w_curr = q_align * sv.world_pos + t_align;
            if !w_curr.is_finite() {
                continue;
            }
            let (gx, gy, gz) = Self::world_to_cell(w_curr);
            let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
            if let Some(chunk) = self.get_or_alloc_chunk(ckey, gen) {
                let idx = VoxelChunk::idx(lx, ly, lz);
                let sdf_norm = (sv.tsdf as f32 / 127.0).clamp(-1.0, 1.0);
                let prior_w = (sv.weight as f32).clamp(MESH_MIN_WEIGHT as f32 + 10.0, 80.0);
                chunk.fuse_voxel(idx, sdf_norm, prior_w, 90, false, gen);
                chunk.dirty = true;
                imported_voxels += 1;
            }
        }
        if imported_voxels >= 16 {
            self.has_dense_depth = true;
        }
        self.step_greedy_meshing(96);
        self.mesh_gen = self.mesh_gen.wrapping_add(1);
    }

    pub fn build_obj_mesh_groups(&mut self) -> Vec<(&'static str, Vec<TriMeshVertex>)> {
        self.step_greedy_meshing(MAX_ACTIVE_CHUNKS);
        let mut detail = Vec::new();
        let mut flat = Vec::new();
        for chunk in self.chunks.values() {
            if !chunk.detail_tris.is_empty() {
                detail.extend_from_slice(&chunk.detail_tris);
            }
            if !chunk.flat_tris.is_empty() {
                flat.extend_from_slice(&chunk.flat_tris);
            }
        }
        let mut groups = Vec::new();
        if !detail.is_empty() {
            groups.push(("DetailObjects", detail));
        }
        if !flat.is_empty() {
            groups.push(("FlatSurfaces", flat));
        }
        groups
    }
}

pub struct TriMeshRenderer {
    program: glow::Program,
    depth_program: glow::Program,
    vao: glow::VertexArray,
    depth_vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    u_depth_mvp: Option<glow::UniformLocation>,
    u_normal_bias: Option<glow::UniformLocation>,
    uploaded_gen: u64,
    vert_count: i32,
}

impl TriMeshRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, TRI_MESH_VERT, TRI_MESH_FRAG)?;
            let depth_program = compile_program(gl, DEPTH_OCC_VERT, DEPTH_OCC_FRAG)?;
            let vao = gl.create_vertex_array()?;
            let depth_vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;

            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                (MAX_MESH_VERTICES * size_of::<TriMeshVertex>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<TriMeshVertex>() as i32;

            gl.bind_vertex_array(Some(vao));
            if let Some(loc) = gl.get_attrib_location(program, "a_pos") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, stride, 0);
            }
            if let Some(loc) = gl.get_attrib_location(program, "a_normal") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, stride, 12);
            }
            if let Some(loc) = gl.get_attrib_location(program, "a_bary") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, stride, 24);
            }
            if let Some(loc) = gl.get_attrib_location(program, "a_color") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 4, glow::FLOAT, false, stride, 36);
            }

            gl.bind_vertex_array(Some(depth_vao));
            if let Some(loc) = gl.get_attrib_location(depth_program, "a_pos") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, stride, 0);
            }
            if let Some(loc) = gl.get_attrib_location(depth_program, "a_normal") {
                gl.enable_vertex_attrib_array(loc);
                gl.vertex_attrib_pointer_f32(loc, 3, glow::FLOAT, false, stride, 12);
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            let u_depth_mvp = gl.get_uniform_location(depth_program, "u_mvp");
            let u_normal_bias = gl.get_uniform_location(depth_program, "u_normal_bias");
            Ok(Self {
                program,
                depth_program,
                vao,
                depth_vao,
                vbo,
                u_mvp,
                u_depth_mvp,
                u_normal_bias,
                uploaded_gen: 0,
                vert_count: 0,
            })
        }
    }

    pub fn sync_if_dirty(&mut self, gl: &glow::Context, verts: &[TriMeshVertex], gen: u64) {
        if gen == self.uploaded_gen {
            return;
        }
        self.uploaded_gen = gen;
        let count = verts.len().min(MAX_MESH_VERTICES);
        self.vert_count = count as i32;
        if count > 0 {
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    verts.as_ptr() as *const u8,
                    count * size_of::<TriMeshVertex>(),
                );
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
                gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
            }
        }
    }

    pub fn draw_depth_occlusion(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.vert_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.color_mask(false, false, false, false);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);

            gl.use_program(Some(self.depth_program));
            gl.bind_vertex_array(Some(self.depth_vao));
            if let Some(loc) = &self.u_depth_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }
            if let Some(loc) = &self.u_normal_bias {
                gl.uniform_1_f32(Some(loc), 0.014);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.vert_count);

            gl.bind_vertex_array(None);
            gl.color_mask(true, true, true, true);
        }
    }

    pub fn draw(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.vert_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.vert_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.vbo);
            gl.delete_vertex_array(self.vao);
            gl.delete_vertex_array(self.depth_vao);
            gl.delete_program(self.program);
            gl.delete_program(self.depth_program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_synthetic_depth_image(dw: usize, dh: usize, depth_fn: impl Fn(usize, usize) -> u16) -> Vec<u8> {
        let mut buf = vec![0u8; dw * dh * 2];
        for y in 0..dh {
            for x in 0..dw {
                let mm = depth_fn(x, y);
                let off = (y * dw + x) * 2;
                let le = mm.to_le_bytes();
                buf[off] = le[0];
                buf[off + 1] = le[1];
            }
        }
        buf
    }

    #[test]
    fn flat_wall_merges_into_few_unified_quads_and_floor_never_renders() {
        let mut map = SpatialMeshMap::default();
        // Register invisible floor at y = -0.80m for physics only
        map.ingest_ar_plane_polygons(
            &[RawArPlane {
                raw_type: 0,
                pos: Vec3::new(0.0, -0.80, -1.2),
                rot: Quat::IDENTITY,
                extent_x: 3.0,
                extent_z: 3.0,
                polygon_xz: vec![[-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5]],
            }],
            0.0,
        );
        assert_eq!(map.inferred_floor_y(), Some(-0.80));

        // Purely flat vertical wall at z = -1.40m (NO bumps!)
        let dw = 80usize;
        let dh = 60usize;
        let depth_wall = make_synthetic_depth_image(dw, dh, |_, _| 1400u16);
        let conf = vec![190u8; dw * dh];
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth_wall, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::ZERO, Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);
        let mut verts = Vec::new();
        let (_, detail_tris, flat_tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert!(flat_tris >= 2, "flat wall must emit merged flat sheets, got {}", flat_tris);
        assert!(
            flat_tris < 90,
            "coplanar merge must collapse flat wall massively (not 1000+ micro-tris!), got flat={}, detail={}",
            flat_tris, detail_tris
        );
        // Verify ZERO floor triangles exist!
        let floor_tris = verts
            .chunks_exact(3)
            .filter(|tri| tri[0].normal[1] >= 0.78 && (tri[0].pos[1] + tri[1].pos[1] + tri[2].pos[1]) / 3.0 <= -0.74)
            .count();
        assert_eq!(floor_tris, 0, "floor must NEVER be rendered in visual mesh!");
    }

    #[test]
    fn closer_scan_refines_bumpy_object_to_finer_triangles_and_instantly_erases_floating_errors() {
        let mut map = SpatialMeshMap::default();
        let dw = 80usize;
        let dh = 60usize;
        let conf = vec![190u8; dw * dh];

        // 1. Scan bumpy curved object from FAR (camera at z = +0.55m -> object at z = -1.0m is 1.55m away -> 5cm LOD)
        let depth_far_bumpy = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - 40.0;
            let dy = y as f32 - 30.0;
            if dx.abs() < 16.0 && dy.abs() < 14.0 {
                // Rippled/bumpy 3D object around 1.55m away
                (1520.0 + ((dx * 0.45).sin() * 45.0 + (dy * 0.45).cos() * 45.0).abs()) as u16
            } else {
                2400u16
            }
        });
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth_far_bumpy, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::new(0.0, 0.0, 0.55), Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);
        let mut verts_far = Vec::new();
        let (_, detail_far, _) = map.build_triangle_mesh_buffer(Vec3::new(0.0, 0.0, 0.55), Vec3::NEG_Z, &mut verts_far);
        assert!(detail_far >= 10, "far bumpy object should emit medium 5cm detail tris, got {}", detail_far);

        // 2. Move camera UP CLOSE (z = -0.35m -> object at z = -1.0m is now 0.65m <= 0.95m -> unlocks ultra-fine 2.5cm tris!)
        let depth_close_bumpy = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - 40.0;
            let dy = y as f32 - 30.0;
            if dx.abs() < 26.0 && dy.abs() < 22.0 {
                (620.0 + ((dx * 0.35).sin() * 38.0 + (dy * 0.35).cos() * 38.0).abs()) as u16
            } else {
                1500u16
            }
        });
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth_close_bumpy, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::new(0.0, 0.0, -0.35), Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);
        let mut verts_close = Vec::new();
        let (_, detail_close, _) = map.build_triangle_mesh_buffer(Vec3::new(0.0, 0.0, -0.35), Vec3::NEG_Z, &mut verts_close);
        assert!(
            detail_close > detail_far,
            "moving camera closer must refine bumpy object into finer/more triangles: far={}, close={}",
            detail_far, detail_close
        );

        // 3. Floating error cleanup: now look straight through the old close object to a wall 0.5m behind it ->
        //    zero carving dead-zone + hard erosion MUST erase it in just 2 Turbo frames!
        let depth_clear = make_synthetic_depth_image(dw, dh, |_, _| 1500u16);
        for _ in 0..2 {
            map.ingest_depth_map(
                &depth_clear, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::new(0.0, 0.0, -0.35), Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);
        let mut verts_healed = Vec::new();
        map.build_triangle_mesh_buffer(Vec3::new(0.0, 0.0, -0.35), Vec3::NEG_Z, &mut verts_healed);
        let remaining_err = verts_healed
            .chunks_exact(3)
            .filter(|tri| {
                let mid = (Vec3::from_array(tri[0].pos) + Vec3::from_array(tri[1].pos) + Vec3::from_array(tri[2].pos)) / 3.0;
                mid.z > -1.20 && mid.x.abs() < 0.25 && mid.y.abs() < 0.22
            })
            .count();
        assert!(
            remaining_err <= 6,
            "zero-dead-zone carving must heal floating errors in 2 frames, remaining={}",
            remaining_err
        );
    }

    #[test]
    fn true_lock_ignores_wobble_and_unscanned_directions_and_requires_3_ticks_for_change() {
        let mut map = SpatialMeshMap::default();
        let dw = 80usize;
        let dh = 60usize;
        let conf = vec![190u8; dw * dh];
        let depth_obj = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - 40.0;
            let dy = y as f32 - 30.0;
            if dx.abs() < 16.0 && dy.abs() < 14.0 {
                (900.0 + (dx * dx) * 0.40) as u16
            } else {
                2000u16
            }
        });
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth_obj, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::ZERO, Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);

        // Unchanged scene with +5cm wobble while Locked -> 0 writes!
        let depth_wobble = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - 40.0;
            let dy = y as f32 - 30.0;
            if dx.abs() < 16.0 && dy.abs() < 14.0 {
                (950.0 + (dx * dx) * 0.40) as u16
            } else {
                2050u16
            }
        });
        let st_w = map.ingest_depth_map(
            &depth_wobble, dw, dh, dw * 2, 2,
            Some(&conf), dw, dh, dw, 1, false,
            64.0, 64.0, 40.0, 30.0, dw, dh,
            Vec3::ZERO, Quat::IDENTITY, false, 2,
        );
        assert_eq!(st_w.updated_voxels, 0);

        // Turning +90 deg into unscanned space while Locked -> 0 writes!
        let st_u = map.ingest_depth_map(
            &depth_obj, dw, dh, dw * 2, 2,
            Some(&conf), dw, dh, dw, 1, false,
            64.0, 64.0, 40.0, 30.0, dw, dh,
            Vec3::ZERO, Quat::from_rotation_y(std::f32::consts::FRAC_PI_2), false, 2,
        );
        assert_eq!(st_u.updated_voxels, 0);

        // 1-frame glitch -> 0 writes!
        let depth_empty = make_synthetic_depth_image(dw, dh, |_, _| 2000u16);
        let st_g1 = map.ingest_depth_map(
            &depth_empty, dw, dh, dw * 2, 2,
            Some(&conf), dw, dh, dw, 1, false,
            64.0, 64.0, 40.0, 30.0, dw, dh,
            Vec3::ZERO, Quat::IDENTITY, false, 2,
        );
        assert_eq!(st_g1.updated_voxels, 0);
    }

    #[test]
    fn scanned_raycast_and_collision_only_hit_scanned_mesh_and_ignore_unscanned_space() {
        let mut map = SpatialMeshMap::default();
        // Empty map -> zero fake hits or collisions!
        assert!(map.raycast_scanned_surface(Vec3::ZERO, Vec3::NEG_Z, 0.15, 4.5).is_none());
        assert!(map.sample_scanned_collision(Vec3::new(0.0, 0.0, -1.36), 0.048).is_none());

        // Scan a real flat vertical wall at z = -1.40m
        let dw = 80usize;
        let dh = 60usize;
        let depth_wall = make_synthetic_depth_image(dw, dh, |_, _| 1400u16);
        let conf = vec![190u8; dw * dh];
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth_wall, dw, dh, dw * 2, 2,
                Some(&conf), dw, dh, dw, 1, false,
                64.0, 64.0, 40.0, 30.0, dw, dh,
                Vec3::ZERO, Quat::IDENTITY, true, 1,
            );
        }
        map.step_greedy_meshing(64);

        // Ray straight at the scanned wall MUST hit near z = -1.40m with +Z outward normal
        let (hit_pos, hit_n, hit_dist) = map
            .raycast_scanned_surface(Vec3::ZERO, Vec3::NEG_Z, 0.15, 4.5)
            .expect("must hit scanned wall");
        assert!((hit_dist - 1.40).abs() < 0.04, "hit_dist={}", hit_dist);
        assert!((hit_pos.z - (-1.40)).abs() < 0.04);
        assert!(hit_n.z > 0.75, "normal should point toward camera (+Z), got {:?}", hit_n);

        // Sphere touching the scanned wall at z = -1.36m (4cm from wall < 4.8cm radius) MUST collide
        let (col_n, pen) = map
            .sample_scanned_collision(Vec3::new(0.0, 0.0, -1.36), 0.048)
            .expect("must collide with scanned wall");
        assert!(col_n.z > 0.75);
        assert!(pen > 0.0);

        // Ray & sphere in unscanned direction (+X) MUST return None!
        assert!(map.raycast_scanned_surface(Vec3::ZERO, Vec3::X, 0.15, 4.5).is_none());
        assert!(map.sample_scanned_collision(Vec3::new(1.5, 0.0, 0.0), 0.048).is_none());
    }
}
