use crate::passthrough::compile_program;
use crate::room_scanner::{
    classify_ar_plane, infer_floor_y, AabbCollider, PlaneMarker, SurfaceKind,
};
use glam::{Mat3, Mat4, Quat, Vec3};
use glow::HasContext;
use std::collections::HashMap;
use std::mem::size_of;

pub const CHUNK_DIM: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM; // 4096 voxels/chunk
pub const CELL_SIZE: f32 = 0.035; // 3.5cm TSDF grid -> 0.56m cube/chunk, sub-cm surface precision via zero-crossing interpolation
pub const TSDF_TRUNC: f32 = CELL_SIZE * 3.0; // ±10.5cm signed distance band around real object surfaces
pub const MAX_ACTIVE_CHUNKS: usize = 256; // Hard RAM cap (~2.0 MB raw TSDF + cached surface tris)
pub const MAX_CANONICAL_PLANES: usize = 32;
pub const MAX_POLY_VERTS: usize = 24;
pub const MAX_MESH_VERTICES: usize = 90_000; // 30,000 crisp conformal triangles (~4.68 MB VBO, sorted nearest-first)
pub const MAX_RENDER_RADIUS_M: f32 = 6.0;

pub const MESH_MIN_WEIGHT: u8 = 28; // Locks in 2 frames high-conf / 3-4 frames smooth anchored interior; rejects 1-frame noise
pub const LOCK_CHANGE_CONFIRM: u8 = 3; // Locked mode requires 3 consecutive ticks (~1.5s) of real change before mutating TSDF

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
    float shade = 0.52 + 0.48 * clamp(dot(n, normalize(vec3(0.32, 0.86, 0.40))) * 0.5 + 0.5, 0.0, 1.0);
    vec3 fill_rgb = v_color.rgb * shade;
    vec3 wire_rgb = mix(v_color.rgb, vec3(1.0), 0.46);

    float fill_a = v_color.a;
    float wire_a = clamp(v_color.a * 2.4 + 0.46, 0.52, 0.90);

    vec3 out_rgb = mix(fill_rgb, wire_rgb, edge_factor);
    float out_a = mix(fill_a, wire_a, edge_factor);
    frag_color = vec4(out_rgb, out_a);
}
"#;

// Pass 0 Hardware Depth-Buffer Pre-Pass: pushes static real-world TSDF surfaces back by `u_normal_bias` (1.4cm)
// so objects resting ON top of desks/floors never clip while everything BEHIND real furniture/mugs/walls
// (both occluded back-scan mesh layers AND virtual 3D balls/cubes/drawings) is cleanly hidden by Z-buffer.
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
    tsdf: Box<[i8; CHUNK_VOLUME]>,        // Quantized [-127..127] => [-1.0..+1.0] (+ = air, 0 = surface, - = inside)
    weight: Box<[u8; CHUNK_VOLUME]>,      // Multi-view fusion confidence weight (0 = unobserved, >= MESH_MIN_WEIGHT = locked)
    change_hits: Box<[u8; CHUNK_VOLUME]>, // Consecutive diff votes in Locked/Maintain mode (must reach LOCK_CHANGE_CONFIRM)
    surface_count: u16,                   // Approximate count of locked near-surface voxels
    dirty: bool,
    last_touched_gen: u64,
    tris: Vec<TriMeshVertex>,
    aabbs: Vec<AabbCollider>,
}

impl VoxelChunk {
    fn new(gen: u64) -> Self {
        Self {
            tsdf: Box::new([127i8; CHUNK_VOLUME]),
            weight: Box::new([0u8; CHUNK_VOLUME]),
            change_hits: Box::new([0u8; CHUNK_VOLUME]),
            surface_count: 0,
            dirty: false,
            last_touched_gen: gen,
            tris: Vec::new(),
            aabbs: Vec::new(),
        }
    }

    #[inline]
    fn idx(lx: usize, ly: usize, lz: usize) -> usize {
        (ly * CHUNK_DIM + lz) * CHUNK_DIM + lx
    }

    #[inline]
    fn is_near_surface(tsdf: i8, weight: u8) -> bool {
        weight >= MESH_MIN_WEIGHT && (-100..=28).contains(&tsdf)
    }

    /// Bit-packed consecutive-frame vote tracker in 1 byte (`[5-bit tick_tag | 3-bit vote_count]`):
    /// - Deduplicates multiple rays hitting the same voxel within the SAME frame (`tick_tag == cur_tag`).
    /// - Increments only across distinct consecutive frames (`diff == 1 || diff == 2`), resetting on long gaps!
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
            return prev_cnt; // Already voted this frame -> never double-count adjacent rays!
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

    #[inline]
    fn fuse_voxel(&mut self, idx: usize, sdf_norm: f32, add_w: f32, gen: u64) -> bool {
        let old_t = self.tsdf[idx];
        let old_w = self.weight[idx];
        let was_surf = Self::is_near_surface(old_t, old_w);

        let old_wf = old_w as f32;
        let new_wf = (old_wf + add_w).min(100.0);
        let old_tf = old_t as f32 / 127.0;
        let fused = if old_w == 0 {
            sdf_norm
        } else {
            (old_tf * old_wf + sdf_norm * add_w) / (old_wf + add_w)
        };

        let new_t = (fused * 127.0).round().clamp(-127.0, 127.0) as i8;
        let new_w = new_wf.round().clamp(1.0, 100.0) as u8;

        self.tsdf[idx] = new_t;
        self.weight[idx] = new_w;
        self.change_hits[idx] = 0;
        self.last_touched_gen = gen;

        let now_surf = Self::is_near_surface(new_t, new_w);
        if !was_surf && now_surf {
            self.surface_count = self.surface_count.saturating_add(1);
        } else if was_surf && !now_surf {
            self.surface_count = self.surface_count.saturating_sub(1);
        }

        // Only mark chunk dirty when lock status, sign, or significant surface displacement changes
        let lock_changed = (old_w < MESH_MIN_WEIGHT) != (new_w < MESH_MIN_WEIGHT);
        let sign_changed = (old_t > 0) != (new_t > 0);
        let shifted = new_w >= MESH_MIN_WEIGHT && (new_t as i16 - old_t as i16).abs() >= 6;
        if lock_changed || sign_changed || shifted {
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
    pub has_seen_conf: bool,
    pub tick_gen: u64,
    pub mesh_gen: u64,
}

impl Default for SpatialMeshMap {
    fn default() -> Self {
        Self {
            planes: Vec::with_capacity(MAX_CANONICAL_PLANES),
            chunks: HashMap::with_capacity(96),
            has_dense_depth: false,
            has_seen_conf: false,
            tick_gen: 0,
            mesh_gen: 1,
        }
    }
}

fn surface_rgb(kind: SurfaceKind) -> [f32; 3] {
    match kind {
        SurfaceKind::Floor => [0.16, 0.86, 0.92],
        SurfaceKind::Wall => [0.34, 0.68, 1.00],
        SurfaceKind::Table => [1.00, 0.72, 0.24],
        SurfaceKind::Seat => [0.76, 0.52, 1.00],
        SurfaceKind::Ceiling => [0.58, 0.65, 0.96],
        SurfaceKind::VerticalObject => [0.52, 0.78, 0.92],
        SurfaceKind::Unknown => [0.75, 0.85, 0.92],
    }
}

/// Colors Tier-2 high-detail TSDF object triangles clearly by 3D curvature so intricate/curved objects
/// stand out sharply from smooth object faces and from Tier-1 cool cyan/blue flat Floors & Walls!
fn tsdf_detail_color(curv: f32, normal: Vec3) -> [f32; 4] {
    if curv >= 0.095 {
        // High-detail edges, curved objects, mugs, laptop bezels, chair contours: vivid amber-gold
        let t = ((curv - 0.095) / 0.22).clamp(0.0, 1.0);
        [1.00, 0.72 - 0.24 * t, 0.18 + 0.08 * (1.0 - t), 0.26]
    } else if normal.y >= 0.78 {
        // Elevated horizontal detail tops (desk top, book, keyboard, chair seat): warm mint-gold
        [0.42, 0.94, 0.56, 0.21]
    } else {
        // Smooth detail object sides / protruding entities: crisp emerald-teal
        [0.12, 0.90, 0.68, 0.20]
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

/// Computes a deduplicated CCW 2D Convex Hull using Andrew's Monotone Chain, capped to `MAX_POLY_VERTS`.
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
fn read_depth_mm(
    depth_bytes: &[u8],
    x: usize,
    y: usize,
    row_stride: usize,
    pix_stride: usize,
) -> u16 {
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

/// Checks if a smooth dense-depth pixel sits inside a REAL object anchored to confident stereo edges
/// (sampling a fast ±2px / ±4px cross in `RawDepthConfidenceImage`), while still rejecting empty mid-air voids!
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

impl SpatialMeshMap {
    pub fn clear(&mut self) {
        self.planes.clear();
        self.chunks.clear();
        self.has_dense_depth = false;
        self.has_seen_conf = false;
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
            .filter(|c| c.surface_count > 0 || !c.tris.is_empty())
            .count()
    }

    pub fn confirmed_voxel_count(&self) -> usize {
        self.chunks.values().map(|c| c.surface_count as usize).sum()
    }

    pub fn canonical_plane_count(&self) -> usize {
        self.planes.len()
    }

    pub fn has_tsdf_mesh(&self) -> bool {
        self.chunks.values().any(|c| !c.tris.is_empty())
    }

    pub fn inferred_floor_y(&self) -> Option<f32> {
        self.planes
            .iter()
            .filter(|p| p.kind == SurfaceKind::Floor)
            .map(|p| p.pos.y)
            .reduce(f32::min)
    }

    pub fn classify_surface_point(&self, pos: Vec3, normal: Vec3, cam_y: f32) -> SurfaceKind {
        let n = normal.normalize_or_zero();
        if !pos.is_finite() || n.length_squared() < 1e-4 {
            return SurfaceKind::Unknown;
        }
        for p in &self.planes {
            if p.normal.dot(n) < 0.80 {
                continue;
            }
            let rel = pos - p.pos;
            if rel.dot(p.normal).abs() > 0.12 {
                continue;
            }
            let lx = rel.dot(p.rot * Vec3::X);
            let lz = rel.dot(p.rot * Vec3::Z);
            if lx.abs() <= p.extent_x * 0.55 + 0.10 && lz.abs() <= p.extent_z * 0.55 + 0.10 {
                return p.kind;
            }
        }
        let raw_type = if n.y >= 0.65 {
            0
        } else if n.y <= -0.65 {
            2
        } else {
            1
        };
        let ext = if raw_type == 1 { 0.85 } else { 0.65 };
        classify_ar_plane(raw_type, pos.y, ext, ext, cam_y, self.inferred_floor_y())
    }

    /// Synthesizes a 36cm x 36cm support patch ONLY on explicit user drop onto a real `Point` normal.
    pub fn ensure_explicit_drop_patch(&mut self, pos: Vec3, normal: Vec3, cam_y: f32) -> SurfaceKind {
        let n = normal.normalize_or_zero();
        if !pos.is_finite() || n.length_squared() < 1e-4 {
            return SurfaceKind::Unknown;
        }
        let kind = self.classify_surface_point(pos, n, cam_y);
        if kind == SurfaceKind::Unknown {
            return kind;
        }
        let already_covered = self.planes.iter().any(|old| {
            if old.normal.dot(n) < 0.85 {
                return false;
            }
            let rel = pos - old.pos;
            if rel.dot(old.normal).abs() > 0.10 {
                return false;
            }
            let lx = rel.dot(old.rot * Vec3::X).abs();
            let lz = rel.dot(old.rot * Vec3::Z).abs();
            lx <= old.extent_x * 0.52 && lz <= old.extent_z * 0.52
        });
        if !already_covered {
            let ref_axis = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
            let tx = ref_axis.cross(n).normalize_or_zero();
            let tz = tx.cross(n).normalize_or_zero();
            let rot = Quat::from_mat3(&Mat3::from_cols(tx, n, tz)).normalize();
            let raw_type = if n.y >= 0.65 {
                0
            } else if n.y <= -0.65 {
                2
            } else {
                1
            };
            self.push_or_evict_plane(CanonicalPlane {
                kind,
                raw_type,
                pos,
                rot,
                normal: n,
                polygon_xz: rect_fallback_poly(0.36, 0.36),
                extent_x: 0.36,
                extent_z: 0.36,
                area: 0.13,
                weight: 1.0,
                last_seen_gen: self.tick_gen,
                is_explicit_patch: true,
            });
            self.mesh_gen = self.mesh_gen.wrapping_add(1);
        }
        kind
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

    /// Rule 1: Coplanar Plane Merging — fuses incoming ARCore polygons for crisp physics colliders,
    /// reticle classification, and non-Depth device fallback (NEVER stacked visually over TSDF mesh!).
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
                let ndot = old.normal.dot(normal);
                if ndot < 0.90 {
                    return false;
                }
                let perp = (raw.pos - old.pos).dot(old.normal).abs();
                if perp > 0.075 {
                    return false;
                }
                let dist = (raw.pos - old.pos).length();
                dist <= old.bounding_radius() + new_rad + 0.30
            });

            if let Some(idx) = match_idx {
                self.merge_into_plane(
                    idx, raw.raw_type, raw.pos, rot, &raw_poly, area, cam_y, floor_y,
                );
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
                });
                changed = true;
            }
        }

        if changed {
            self.coalesce_overlapping_planes(cam_y, floor_y);
            // Mark chunks touching structural Floor/Wall/Ceiling planes dirty so flat micro-tris yield cleanly to Tier-1 planes
            for (&ckey, chunk) in &mut self.chunks {
                if chunk.tris.is_empty() {
                    continue;
                }
                let span = CHUNK_DIM as f32 * CELL_SIZE;
                let cc = Vec3::new(
                    (ckey.0 as f32 + 0.5) * span,
                    (ckey.1 as f32 + 0.5) * span,
                    (ckey.2 as f32 + 0.5) * span,
                );
                let touches_struct = self.planes.iter().any(|p| {
                    matches!(p.kind, SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Ceiling)
                        && (cc - p.pos).dot(p.normal).abs() <= span * 0.70
                });
                if touches_struct {
                    chunk.dirty = true;
                }
            }
            self.mesh_gen = self.mesh_gen.wrapping_add(1);
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

        let mut combined_xz: Vec<[f32; 2]> =
            Vec::with_capacity(old.polygon_xz.len() + new_poly.len());
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
            old.kind =
                classify_ar_plane(raw_type, old.pos.y, old.extent_x, old.extent_z, cam_y, floor_y);
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
            let dx_range = if lx == 0 {
                -1..=0
            } else if lx == LAST {
                0..=1
            } else {
                0..=0
            };
            let dy_range = if ly == 0 {
                -1..=0
            } else if ly == LAST {
                0..=1
            } else {
                0..=0
            };
            let dz_range = if lz == 0 {
                -1..=0
            } else if lz == LAST {
                0..=1
            } else {
                0..=0
            };
            for dx in dx_range {
                for dy in dy_range.clone() {
                    for dz in dz_range.clone() {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        if let Some(nb) =
                            self.chunks.get_mut(&(ckey.0 + dx, ckey.1 + dy, ckey.2 + dz))
                        {
                            nb.dirty = true;
                        }
                    }
                }
            }
        }
    }

    /// Integrates a single surface ray `(cam_pos, dir, d_obs)` across the TSDF truncation band
    /// `[d_obs - TSDF_TRUNC, d_obs + 0.72 * TSDF_TRUNC]`.
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
        let t_start = (d_obs - TSDF_TRUNC * 0.92).max(0.18);
        let t_end = d_obs + TSDF_TRUNC * 0.72;
        let step = CELL_SIZE * 0.80; // 2.8cm step (< 3.5cm cell) guarantees continuous zero-crossing band

        let mut prev_cell = (i32::MAX, i32::MAX, i32::MAX);
        let mut updated = 0u32;
        let mut t = t_start;
        while t <= t_end {
            let p = cam_pos + dir * t;
            let cell = Self::world_to_cell(p);
            if cell != prev_cell {
                prev_cell = cell;
                let center = Self::cell_center(cell.0, cell.1, cell.2);
                let ray_t = (center - cam_pos).dot(dir);
                let sdf = ((d_obs - ray_t) / TSDF_TRUNC).clamp(-1.0, 1.0);
                let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
                if let Some(chunk) = self.get_or_alloc_chunk(ckey, gen) {
                    let idx = VoxelChunk::idx(lx, ly, lz);
                    if chunk.fuse_voxel(idx, sdf, w, gen) {
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

    /// Checks if the observed depth `d_obs` already matches an existing locked surface within `±11cm`.
    /// Resets `change_hits` on the matched surface so sporadic single-frame noise can never accumulate.
    fn match_locked_surface_along_ray(&mut self, cam_pos: Vec3, dir: Vec3, d_obs: f32) -> bool {
        let t_min = (d_obs - 0.11).max(0.20);
        let t_max = d_obs + 0.11;
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
                        if chunk.weight[idx] >= MESH_MIN_WEIGHT
                            && (-105..=40).contains(&chunk.tsdf[idx])
                        {
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

    /// Checks if a genuine locked surface exists BEHIND `d_obs` (`d_obs + 0.14m .. d_obs + 1.8m`),
    /// proving a newly detected surface in Locked/Maintain mode is inside our mapped room rather than unscanned space.
    fn has_locked_surface_behind(&self, cam_pos: Vec3, dir: Vec3, d_obs: f32) -> bool {
        let t_start = d_obs + 0.14;
        let t_end = (d_obs + 1.80).min(3.15);
        let step = CELL_SIZE * 1.35;
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

    /// Free-space ray carving strictly IN FRONT of the observed surface.
    /// - Turbo (`diff_only = false`): carves immediately (`< CELL_SIZE` step, chunk-miss cached).
    /// - Locked/Maintain (`diff_only = true`): increments `change_hits` and ONLY carves once a voxel
    ///   has been persistently seen-through for `LOCK_CHANGE_CONFIRM` (3+) consecutive ticks (~1.5s)!
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
        let margin = if diff_only { 0.15 } else { TSDF_TRUNC * 1.30 };
        let max_t = (d_obs - margin).min(2.6);
        if max_t <= 0.24 {
            return 0;
        }
        let step = CELL_SIZE * 0.85; // < CELL_SIZE so carving never skips a voxel along the ray
        let mut t = 0.24f32;
        let mut carved = 0u32;
        let mut steps = 0usize;
        let mut prev_cell = (i32::MAX, i32::MAX, i32::MAX);
        let mut cached_miss_ckey = (i32::MAX, i32::MAX, i32::MAX);

        while t < max_t && steps < 84 {
            let p = cam_pos + dir * t;
            let cell = Self::world_to_cell(p);
            if cell != prev_cell {
                prev_cell = cell;
                let (ckey, lx, ly, lz) = Self::split_cell(cell.0, cell.1, cell.2);
                if ckey != cached_miss_ckey {
                    if let Some(chunk) = self.chunks.get_mut(&ckey) {
                        let idx = VoxelChunk::idx(lx, ly, lz);
                        let old_w = chunk.weight[idx];
                        let old_t = chunk.tsdf[idx];
                        if diff_only {
                            if old_w >= MESH_MIN_WEIGHT && old_t <= 16 {
                                let votes = chunk.vote_change(idx, gen);
                                if votes >= LOCK_CHANGE_CONFIRM {
                                    if chunk.weight[idx] > 32 {
                                        chunk.weight[idx] = 32;
                                    }
                                    if chunk.fuse_voxel(idx, 1.0, carve_w.max(32.0), gen) {
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
                            if chunk.weight[idx] > 34 {
                                chunk.weight[idx] = 34;
                            }
                            if chunk.fuse_voxel(idx, 1.0, carve_w.max(28.0), gen) {
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
                        cached_miss_ckey = ckey;
                    }
                }
            }
            t += step;
            steps += 1;
        }
        carved
    }

    /// Returns `true` if `pos` sits flush (`±4.2cm`) on a Tier-1 flat structural Floor, Wall, or Ceiling plane
    /// (never Tables/Seats/Objects!). Used to separate sleek low-poly walls/floors from high-detail 3.5cm TSDF objects.
    fn is_flush_structural_plane(&self, pos: Vec3, opt_normal: Option<Vec3>) -> bool {
        self.planes.iter().any(|p| {
            if !matches!(
                p.kind,
                SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Ceiling
            ) {
                return false;
            }
            if let Some(n) = opt_normal {
                if n.dot(p.normal) < 0.85 {
                    return false;
                }
            }
            let rel = pos - p.pos;
            if rel.dot(p.normal).abs() > 0.042 {
                return false;
            }
            let lx = rel.dot(p.rot * Vec3::X).abs();
            let lz = rel.dot(p.rot * Vec3::Z).abs();
            lx <= p.extent_x * 0.52 + 0.04 && lz <= p.extent_z * 0.52 + 0.04
        })
    }

    /// Primary Two-Tier Conformal 3D Object Scanner: ingests ARCore 16-bit Dense Depth (`AUTOMATIC` or `RAW_DEPTH_ONLY`)
    /// directly using official `cam.imageIntrinsics` + `cam.pose` sensor-space unprojection.
    ///
    /// - **Two-Tier Region Split:** Flat low-curvature pixels flush on canonical Floors/Walls/Ceilings yield to sleek
    ///   Tier-1 architectural planes (while still carving empty air in front); protruding / high-curvature object regions
    ///   (cups, laptops, chairs, tables, corners) fuse into crisp `3.5cm` Tier-2 TSDF Surface Nets!
    /// - **Neighborhood-Anchored Soft Confidence:** accepts smooth object interiors anchored within `±4px` to confident
    ///   stereo edges (`>=35`) while rejecting unanchored `conf=0` mid-air voids!
    /// - **`turbo == false` (`🔒 Khóa`)**: TRUE LOCK — ignores `±11cm` depth wobble, never scans unscanned directions,
    ///   and requires 3 consecutive confirmed change ticks before mutating TSDF.
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

        if conf_bytes.is_some() {
            self.has_seen_conf = true;
        }

        self.tick_gen = self.tick_gen.wrapping_add(1);
        let gen = self.tick_gen;
        let rot = cam_rot.normalize();

        let sx = dw as f32 / img_w as f32;
        let sy = dh as f32 / img_h as f32;
        let fx_d = fx * sx;
        let fy_d = fy * sy;
        let cx_d = cx * sx;
        let cy_d = cy * sy;

        let step_px = stride.clamp(1, 4);
        let pad_x = (dw / 24).max(2);
        let pad_y = (dh / 24).max(2);
        let max_depth_m = if turbo { 2.95f32 } else { 2.75f32 };

        let mut stats = DepthIngestStats::default();
        let mut border_flips: Vec<((i32, i32, i32), usize, usize, usize)> = Vec::with_capacity(256);

        let mut y = pad_y;
        while y + pad_y < dh {
            let mut x = pad_x;
            while x + pad_x < dw {
                let d_mm = read_depth_mm(depth_bytes, x, y, d_row_stride, d_pix_stride);
                if (260..=2950).contains(&d_mm) {
                    let dm = d_mm as f32 * 0.001;
                    if dm <= max_depth_m {
                        // Neighborhood-anchored soft confidence: wraps real smooth objects cleanly while blocking mid-air voids!
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
                                    (true, if c0 >= 130 { 18.0 } else { 13.0 }, c0 >= 85)
                                } else {
                                    (false, 0.0f32, false)
                                }
                            } else if c0 >= 95 {
                                (true, 18.0f32, true)
                            } else if c0 >= 35 {
                                (true, 12.0f32, true)
                            } else {
                                let nb_c = max_nearby_conf(cb, x, y, dw, dh, cw, ch, c_row_stride, c_pix_stride);
                                if nb_c >= 35 {
                                    // Smooth object interior anchored to nearby confident stereo edge
                                    (true, 9.5f32, nb_c >= 80)
                                } else {
                                    // Unanchored conf=0 void (blank mid-air hallucination) -> REJECT!
                                    (false, 0.0f32, false)
                                }
                            }
                        } else {
                            // Alternate ARCore frames when RawDepthConfidence throws NotYetAvailableException:
                            // still fuse smooth depth gently (needs 3 independent frames to reach MESH_MIN_WEIGHT=28)
                            (true, 10.5f32, true)
                        };

                        if !conf_ok {
                            x += step_px;
                            continue;
                        }

                        // Smart ±1px silhouette jump guard: removes spiderwebs while preserving mugs/bottles/chair legs!
                        let dl1 = read_depth_mm(depth_bytes, x - 1, y, d_row_stride, d_pix_stride);
                        let dr1 = read_depth_mm(depth_bytes, x + 1, y, d_row_stride, d_pix_stride);
                        let du1 = read_depth_mm(depth_bytes, x, y - 1, d_row_stride, d_pix_stride);
                        let dd1 = read_depth_mm(depth_bytes, x, y + 1, d_row_stride, d_pix_stride);
                        if dl1 == 0 || dr1 == 0 || du1 == 0 || dd1 == 0 {
                            x += step_px;
                            continue;
                        }
                        let max_jump = ((d_mm as f32) * 0.085).clamp(55.0, 160.0) as i32;
                        let edge_jump = (dl1 as i32 - d_mm as i32).abs() > max_jump
                            || (dr1 as i32 - d_mm as i32).abs() > max_jump
                            || (du1 as i32 - d_mm as i32).abs() > max_jump
                            || (dd1 as i32 - d_mm as i32).abs() > max_jump;

                        if !edge_jump {
                            // Exact ARCore pinhole unprojection in `cam.pose` sensor frame (+X right, +Y up, -Z forward)
                            let p_cam = Vec3::new(
                                (x as f32 - cx_d) / fx_d * dm,
                                (cy_d - y as f32) / fy_d * dm,
                                -dm,
                            );
                            let world_vec = rot * p_cam;
                            let dist = world_vec.length();
                            if dist >= 0.26 && dist.is_finite() {
                                let dir = world_vec / dist;
                                let hit_world = cam_pos + world_vec;
                                stats.valid_rays += 1;

                                // Local depth 2nd-derivative curvature (mm): separates flat wall/floor patches from detailed objects
                                let curv_mm = (dl1 as i32 + dr1 as i32 - 2 * d_mm as i32).abs()
                                    + (du1 as i32 + dd1 as i32 - 2 * d_mm as i32).abs();
                                let flat_structural = curv_mm < 16
                                    && self.is_flush_structural_plane(hit_world, None);

                                if turbo {
                                    // TURBO CONTINUOUS SCAN: carve empty air along sightline, then fuse HIGH-DETAIL regions into TSDF
                                    stats.changed_rays += 1;
                                    if allow_carve {
                                        stats.updated_voxels += self.carve_free_space_ray(
                                            cam_pos,
                                            dir,
                                            dist,
                                            28.0,
                                            false,
                                            gen,
                                            &mut border_flips,
                                        );
                                    }
                                    if !flat_structural {
                                        stats.updated_voxels += self.integrate_surface_band(
                                            cam_pos,
                                            dir,
                                            dist,
                                            base_w,
                                            gen,
                                            &mut border_flips,
                                        );
                                    }
                                } else {
                                    // TRUE LOCK MODE:
                                    // 1. If depth matches an existing locked TSDF surface (or flat structural plane) -> DO NOTHING!
                                    if flat_structural || self.match_locked_surface_along_ray(cam_pos, dir, dist) {
                                        // Even if hitting a flat wall behind a removed object, still allow 3-tick carving in front!
                                        if flat_structural && allow_carve {
                                            let carved = self.carve_free_space_ray(
                                                cam_pos,
                                                dir,
                                                dist,
                                                34.0,
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
                                    // 2. Check if an old locked object in front of `dist` was removed (needs 3+ consecutive ticks!)
                                    if allow_carve {
                                        let carved = self.carve_free_space_ray(
                                            cam_pos,
                                            dir,
                                            dist,
                                            34.0,
                                            true,
                                            gen,
                                            &mut border_flips,
                                        );
                                        if carved > 0 {
                                            stats.changed_rays += 1;
                                            stats.updated_voxels += carved;
                                        }
                                    }
                                    // 3. ONLY add a new object if a locked surface (TSDF or Floor/Wall) exists BEHIND it
                                    //    AND it stays stationary for 3+ consecutive ticks!
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
                || !c.tris.is_empty()
                || gen.saturating_sub(c.last_touched_gen) <= 45
        });

        stats
    }

    /// Fallback sparse ARCore SLAM PointCloud TSDF ingestion (active only when dense Depth API has not produced data).
    pub fn ingest_point_cloud(&mut self, point_count: usize, packed_xyzc: &[f32], cam_pos: Vec3) {
        if self.has_dense_depth {
            return;
        }
        self.tick_gen = self.tick_gen.wrapping_add(1);
        let gen = self.tick_gen;
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
            let delta = pt - cam_pos;
            let dist_sq = delta.length_squared();
            if !(0.22 * 0.22..=3.5 * 3.5).contains(&dist_sq) {
                continue;
            }
            let dist = dist_sq.sqrt();
            let dir = delta / dist;

            if i % 3 == 0 {
                self.carve_free_space_ray(cam_pos, dir, dist, 12.0, false, gen, &mut border_flips);
            }
            let w = if conf >= 0.72 { 16.0 } else { 12.0 };
            self.integrate_surface_band(cam_pos, dir, dist, w, gen, &mut border_flips);
        }

        if !border_flips.is_empty() {
            self.mark_border_neighbors_dirty(&border_flips);
        }

        self.chunks.retain(|_, c| {
            c.surface_count > 0
                || !c.tris.is_empty()
                || gen.saturating_sub(c.last_touched_gen) <= 60
        });
    }

    /// Budgeted Incremental TSDF Zero-Crossing Surface Nets Meshing — rebuilds at most `max_dirty_chunks` per frame.
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
            let (tris, aabbs) = self.build_chunk_tsdf_surface_nets(*ckey, floor_y);
            if let Some(chunk) = self.chunks.get_mut(ckey) {
                chunk.tris = tris;
                chunk.aabbs = aabbs;
                chunk.dirty = false;
            }
        }

        self.mesh_gen = self.mesh_gen.wrapping_add(1);
        dirty_keys.len()
    }

    /// Fast L1-Cache Halo TSDF Zero-Crossing Surface Nets Mesher:
    /// Copies chunk + 1-voxel neighbor border into an `18×18×18` flat array (at most 27 HashMap lookups),
    /// interpolates exact `tsdf = 0` crossings on all 12 dual-cell edges, computes outward TSDF-gradient normals,
    /// applies gentle 20% Laplacian polish (preserving sharp edges on laptops/mugs/tables), and emits
    /// guaranteed outward-CCW single-sided conformal triangles with ZERO back-face bubbles.
    fn build_chunk_tsdf_surface_nets(
        &self,
        ckey: (i32, i32, i32),
        floor_y: Option<f32>,
    ) -> (Vec<TriMeshVertex>, Vec<AabbCollider>) {
        let Some(chunk) = self.chunks.get(&ckey) else {
            return (Vec::new(), Vec::new());
        };
        if chunk.surface_count == 0 {
            return (Vec::new(), Vec::new());
        }

        let base_gx = ckey.0 * (CHUNK_DIM as i32);
        let base_gy = ckey.1 * (CHUNK_DIM as i32);
        let base_gz = ckey.2 * (CHUNK_DIM as i32);

        // Step 0: Populate 18x18x18 padded primal halo (indices 0..18 map to local -1..=16)
        const H: usize = CHUNK_DIM + 2; // 18
        const H_VOL: usize = H * H * H; // 5832
        let h_idx = |x: usize, y: usize, z: usize| -> usize { (y * H + z) * H + x };

        let mut h_tsdf = [127i8; H_VOL];
        let mut h_weight = [0u8; H_VOL];

        // Copy center chunk interior (local 0..16 -> halo 1..17)
        for ly in 0..CHUNK_DIM {
            for lz in 0..CHUNK_DIM {
                let src_row = (ly * CHUNK_DIM + lz) * CHUNK_DIM;
                let dst_row = ((ly + 1) * H + (lz + 1)) * H + 1;
                h_tsdf[dst_row..dst_row + CHUNK_DIM]
                    .copy_from_slice(&chunk.tsdf[src_row..src_row + CHUNK_DIM]);
                h_weight[dst_row..dst_row + CHUNK_DIM]
                    .copy_from_slice(&chunk.weight[src_row..src_row + CHUNK_DIM]);
            }
        }

        // Copy 1-voxel boundary faces/edges/corners from up to 26 neighbors
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
                            h_weight[dst..dst + sx_len]
                                .copy_from_slice(&nb.weight[src..src + sx_len]);
                        }
                    }
                }
            }
        }

        // Step A: Place dual vertices on 17x17x17 grid at exact linear zero-crossings of the 12 cube edges
        const D: usize = CHUNK_DIM + 1; // 17
        const D_VOL: usize = D * D * D;
        let d_idx = |x: usize, y: usize, z: usize| -> usize { (y * D + z) * D + x };

        let mut dual_pos: Vec<Option<Vec3>> = vec![None; D_VOL];
        let mut dual_norm: Vec<Vec3> = vec![Vec3::Y; D_VOL];

        const CUBE_EDGES: [(usize, usize); 12] = [
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        const CORNER_OFF: [(usize, usize, usize); 8] = [
            (0, 0, 0),
            (1, 0, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 0),
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
        ];

        for ly in 0..D {
            for lz in 0..D {
                for lx in 0..D {
                    let mut c_tsdf = [0i8; 8];
                    let mut c_w = [0u8; 8];
                    let mut c_pos = [Vec3::ZERO; 8];
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
                            if t > 0 {
                                any_pos = true;
                            } else {
                                any_neg = true;
                            }
                        }
                        // Halo index (lx + ox) corresponds to global cell (base_gx + lx + ox - 1)
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

                        // TSDF gradient points from inside (-tsdf) toward empty air (+tsdf) = exact outward normal!
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

        // Step B: Gentle 20% Laplacian polish + local 3D curvature estimation (separates high-detail edges from flat patches)
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
                        (-1isize, 0, 0),
                        (1, 0, 0),
                        (0, -1, 0),
                        (0, 1, 0),
                        (0, 0, -1),
                        (0, 0, 1),
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
                        smooth_pos[di] = Some(cur * 0.80 + (nb_sum / nb_cnt as f32) * 0.20);
                        dual_curv[di] = curv_sum / (nb_cnt as f32);
                    }
                }
            }
        }

        // Step C: Stitch single-sided outward-CCW quads across locked primal sign-changing edges (+X, +Y, +Z)
        let mut tris = Vec::with_capacity((chunk.surface_count as usize) * 6);
        let _ = floor_y;

        for ly in 0..CHUNK_DIM {
            for lz in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    let hi0 = h_idx(lx + 1, ly + 1, lz + 1);
                    let w0 = h_weight[hi0];
                    if w0 < MESH_MIN_WEIGHT {
                        continue;
                    }
                    // Anti-floater dust filter: strip isolated 1-voxel specks (0 locked surface neighbors)
                    let mut nb_surf = 0u8;
                    for (ox, oy, oz) in [
                        (0usize, 1usize, 1usize),
                        (2, 1, 1),
                        (1, 0, 1),
                        (1, 2, 1),
                        (1, 1, 0),
                        (1, 1, 2),
                    ] {
                        let ni = h_idx(lx + ox, ly + oy, lz + oz);
                        if h_weight[ni] >= MESH_MIN_WEIGHT && (-105..=32).contains(&h_tsdf[ni]) {
                            nb_surf += 1;
                        }
                    }
                    if nb_surf < 1 {
                        continue;
                    }
                    let t0 = h_tsdf[hi0];
                    let s0_solid = t0 <= 0;

                    // Edge along +X: voxel (lx, ly, lz) -> (lx+1, ly, lz)
                    let hix = h_idx(lx + 2, ly + 1, lz + 1);
                    if h_weight[hix] >= MESH_MIN_WEIGHT && (h_tsdf[hix] <= 0) != s0_solid {
                        let i0 = d_idx(lx + 1, ly, lz);
                        let i1 = d_idx(lx + 1, ly + 1, lz);
                        let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                        let i3 = d_idx(lx + 1, ly, lz + 1);
                        if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                            (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                        {
                            let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3])
                                .normalize_or_zero();
                            let n = if n.length_squared() > 1e-4 {
                                n
                            } else if s0_solid {
                                Vec3::X
                            } else {
                                Vec3::NEG_X
                            };
                            let mid = (p0 + p2) * 0.5;
                            let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                            if curv >= 0.085 || !self.is_flush_structural_plane(mid, Some(n)) {
                                let col = tsdf_detail_color(curv, n);
                                if s0_solid {
                                    push_mesh_quad(&mut tris, p0, p1, p2, p3, n, col);
                                } else {
                                    push_mesh_quad(&mut tris, p0, p3, p2, p1, n, col);
                                }
                            }
                        }
                    }

                    // Edge along +Y: voxel (lx, ly, lz) -> (lx, ly+1, lz)
                    let hiy = h_idx(lx + 1, ly + 2, lz + 1);
                    if h_weight[hiy] >= MESH_MIN_WEIGHT && (h_tsdf[hiy] <= 0) != s0_solid {
                        let i0 = d_idx(lx, ly + 1, lz);
                        let i1 = d_idx(lx, ly + 1, lz + 1);
                        let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                        let i3 = d_idx(lx + 1, ly + 1, lz);
                        if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                            (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                        {
                            let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3])
                                .normalize_or_zero();
                            let n = if n.length_squared() > 1e-4 {
                                n
                            } else if s0_solid {
                                Vec3::Y
                            } else {
                                Vec3::NEG_Y
                            };
                            let mid = (p0 + p2) * 0.5;
                            let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                            if curv >= 0.085 || !self.is_flush_structural_plane(mid, Some(n)) {
                                let col = tsdf_detail_color(curv, n);
                                if s0_solid {
                                    push_mesh_quad(&mut tris, p0, p1, p2, p3, n, col);
                                } else {
                                    push_mesh_quad(&mut tris, p0, p3, p2, p1, n, col);
                                }
                            }
                        }
                    }

                    // Edge along +Z: voxel (lx, ly, lz) -> (lx, ly, lz+1)
                    let hiz = h_idx(lx + 1, ly + 1, lz + 2);
                    if h_weight[hiz] >= MESH_MIN_WEIGHT && (h_tsdf[hiz] <= 0) != s0_solid {
                        let i0 = d_idx(lx, ly, lz + 1);
                        let i1 = d_idx(lx + 1, ly, lz + 1);
                        let i2 = d_idx(lx + 1, ly + 1, lz + 1);
                        let i3 = d_idx(lx, ly + 1, lz + 1);
                        if let (Some(p0), Some(p1), Some(p2), Some(p3)) =
                            (smooth_pos[i0], smooth_pos[i1], smooth_pos[i2], smooth_pos[i3])
                        {
                            let n = (dual_norm[i0] + dual_norm[i1] + dual_norm[i2] + dual_norm[i3])
                                .normalize_or_zero();
                            let n = if n.length_squared() > 1e-4 {
                                n
                            } else if s0_solid {
                                Vec3::Z
                            } else {
                                Vec3::NEG_Z
                            };
                            let mid = (p0 + p2) * 0.5;
                            let curv = (dual_curv[i0] + dual_curv[i1] + dual_curv[i2] + dual_curv[i3]) * 0.25;
                            if curv >= 0.085 || !self.is_flush_structural_plane(mid, Some(n)) {
                                let col = tsdf_detail_color(curv, n);
                                if s0_solid {
                                    push_mesh_quad(&mut tris, p0, p1, p2, p3, n, col);
                                } else {
                                    push_mesh_quad(&mut tris, p0, p3, p2, p1, n, col);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Step D: Merged per-layer XZ physics AABBs directly on the zero-crossing shell for 120Hz collisions
        let mut aabbs = Vec::with_capacity(16);
        for ly in 0..CHUNK_DIM {
            let mut mask = [[false; CHUNK_DIM]; CHUNK_DIM];
            let mut any = false;
            for lz in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    let idx = VoxelChunk::idx(lx, ly, lz);
                    if chunk.weight[idx] >= MESH_MIN_WEIGHT
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
                    let max =
                        min + Vec3::new(w as f32 * CELL_SIZE, CELL_SIZE, h as f32 * CELL_SIZE);
                    aabbs.push(AabbCollider {
                        min,
                        max,
                        is_object_box: false,
                    });
                    lx += w;
                }
            }
        }

        (tris, aabbs)
    }

    pub fn snapshot_plane_markers(&self, include_all_types: bool) -> Vec<PlaneMarker> {
        self.planes
            .iter()
            .filter(|p| {
                include_all_types
                    || matches!(
                        p.kind,
                        SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Table | SurfaceKind::Seat
                    )
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
                if d2 <= 3.2 * 3.2 {
                    Some((d2, chunk))
                } else {
                    None
                }
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

    /// Builds Two-Tier Triangle Mesh VBO returning `(active_chunks, detail_object_tris, structural_plane_tris)`:
    /// - **Tier 2 (High-Detail Objects):** Nearest-first `3.5cm` TSDF Surface Nets (emerald/gold, up to `82,000` verts).
    /// - **Tier 1 (Flat Structural Floors / Walls / Ceilings):** Clean coarse architectural triangles (cyan/blue, recessed `-6mm`
    ///   for zero Z-fighting where objects touch the floor/wall). Never stacks Table planes over TSDF objects!
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
            if chunk.tris.is_empty() {
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

        // Nearest-first sort ensures objects right in front of the user ALWAYS get full vertex priority!
        visible_chunks.sort_by(|a, b| a.0.total_cmp(&b.0));

        let detail_vert_cap = MAX_MESH_VERTICES.saturating_sub(8_000);
        let mut active_chunks = 0u32;
        for (_, chunk) in visible_chunks {
            active_chunks += 1;
            let remain = detail_vert_cap.saturating_sub(out.len());
            let copy_n = (chunk.tris.len().min(remain) / 3) * 3;
            if copy_n > 0 {
                out.extend_from_slice(&chunk.tris[..copy_n]);
            }
            if out.len() + 3 > detail_vert_cap {
                break;
            }
        }
        let detail_verts = out.len();

        // Tier 1: Always draw clean coarse Floor / Wall / Ceiling structural sheets (zero stacking with Table planes!)
        let include_non_structural_fallback = !self.has_dense_depth && detail_verts < 30;
        for plane in &self.planes {
            let is_structural = matches!(
                plane.kind,
                SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Ceiling
            );
            if !is_structural && !include_non_structural_fallback {
                continue;
            }
            if out.len() + 600 > MAX_MESH_VERTICES {
                break;
            }
            if (plane.pos - cam_pos).length() > MAX_RENDER_RADIUS_M + plane.bounding_radius() {
                continue;
            }
            triangulate_canonical_plane(plane, out);
        }

        let detail_tris = (detail_verts / 3) as u32;
        let plane_tris = ((out.len() - detail_verts) / 3) as u32;
        (active_chunks, detail_tris, plane_tris)
    }

    /// Exports active TSDF narrow-band voxels (`weight >= MESH_MIN_WEIGHT && |tsdf| <= 118`) for `.arroom` v2 persistence.
    pub fn export_tsdf_chunks(&self) -> Vec<((i32, i32, i32), Vec<(u16, i8, u8)>)> {
        let mut out = Vec::new();
        for (&ckey, chunk) in &self.chunks {
            if chunk.surface_count == 0 && chunk.tris.is_empty() {
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
                let prior_w = (sv.weight as f32).clamp(MESH_MIN_WEIGHT as f32 + 12.0, 120.0);
                chunk.fuse_voxel(idx, sdf_norm, prior_w, gen);
                chunk.dirty = true;
                imported_voxels += 1;
            }
        }
        if imported_voxels >= 16 {
            self.has_dense_depth = true;
        }
        self.step_greedy_meshing(64);
        self.mesh_gen = self.mesh_gen.wrapping_add(1);
    }

    pub fn build_obj_mesh_groups(&mut self) -> Vec<(&'static str, Vec<TriMeshVertex>)> {
        self.step_greedy_meshing(MAX_ACTIVE_CHUNKS);
        let mut groups: Vec<(&'static str, Vec<TriMeshVertex>)> = vec![
            ("ConformalSurfaceMesh", Vec::new()),
            ("Floor", Vec::new()),
            ("Table", Vec::new()),
            ("Seat", Vec::new()),
            ("Wall", Vec::new()),
            ("Ceiling", Vec::new()),
        ];
        for chunk in self.chunks.values() {
            if !chunk.tris.is_empty() {
                groups[0].1.extend_from_slice(&chunk.tris);
            }
        }
        if groups[0].1.is_empty() {
            for plane in &self.planes {
                let slot = match plane.kind {
                    SurfaceKind::Floor => 1,
                    SurfaceKind::Table => 2,
                    SurfaceKind::Seat => 3,
                    SurfaceKind::Wall | SurfaceKind::VerticalObject => 4,
                    SurfaceKind::Ceiling => 5,
                    SurfaceKind::Unknown => continue,
                };
                triangulate_canonical_plane(plane, &mut groups[slot].1);
            }
        }
        groups.into_iter().filter(|(_, v)| !v.is_empty()).collect()
    }
}

/// Clean coarse architectural triangulation (`~38cm` ring spacing, subtle cool fill, `-6mm` recessed
/// so Tier-2 high-detail objects sitting on the floor or leaning against walls NEVER Z-fight!).
pub fn triangulate_canonical_plane(plane: &CanonicalPlane, out: &mut Vec<TriMeshVertex>) {
    let n = plane.polygon_xz.len();
    if n < 3 {
        return;
    }
    let ax_x = plane.rot * Vec3::X;
    let ax_z = plane.rot * Vec3::Z;
    let normal = plane.normal;
    let is_structural = matches!(
        plane.kind,
        SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Ceiling
    );
    // Slight -6mm recess on Floor/Wall/Ceiling keeps Tier-2 detail objects visually crisp in front
    let origin = if is_structural {
        plane.pos - normal * 0.006
    } else {
        plane.pos + normal * 0.002
    };
    let rgb = surface_rgb(plane.kind);
    let alpha = match plane.kind {
        SurfaceKind::Floor => 0.085,
        SurfaceKind::Wall => 0.095,
        SurfaceKind::Ceiling => 0.065,
        _ => 0.14,
    };
    let col = [rgb[0], rgb[1], rgb[2], alpha];

    let mut cx = 0.0f32;
    let mut cz = 0.0f32;
    for &[x, z] in &plane.polygon_xz {
        cx += x;
        cz += z;
    }
    cx /= n as f32;
    cz /= n as f32;

    let avg_r: f32 = plane
        .polygon_xz
        .iter()
        .map(|&[x, z]| ((x - cx) * (x - cx) + (z - cz) * (z - cz)).sqrt())
        .sum::<f32>()
        / (n as f32);

    let ring_step = if is_structural { 0.38 } else { 0.24 };
    let rings = ((avg_r / ring_step).ceil() as usize).clamp(1, 4);
    let to_world = |x: f32, z: f32| origin + ax_x * x + ax_z * z;
    let center_w = to_world(cx, cz);

    for r in 0..rings {
        let t0 = (r as f32) / (rings as f32);
        let t1 = ((r + 1) as f32) / (rings as f32);
        for i in 0..n {
            let [x0, z0] = plane.polygon_xz[i];
            let [x1, z1] = plane.polygon_xz[(i + 1) % n];
            let p10 = to_world(cx + (x0 - cx) * t1, cz + (z0 - cz) * t1);
            let p11 = to_world(cx + (x1 - cx) * t1, cz + (z1 - cz) * t1);
            if r == 0 {
                push_mesh_tri(out, center_w, p11, p10, normal, col);
            } else {
                let p00 = to_world(cx + (x0 - cx) * t0, cz + (z0 - cz) * t0);
                let p01 = to_world(cx + (x1 - cx) * t0, cz + (z1 - cz) * t0);
                push_mesh_quad(out, p00, p01, p11, p10, normal, col);
            }
        }
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

    /// Pass 0 Hardware Depth-Buffer Pre-Pass: writes static room surfaces into Z-buffer only
    /// (`color_mask = false`, pushed back `1.4cm` along `-normal` so items sitting ON tables never clip,
    /// while occluded back-scan meshes AND virtual objects behind furniture/mugs/walls vanish cleanly).
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
            // Back-face culling works cleanly because TSDF Surface Nets guarantees outward CCW winding!
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
    fn coplanar_planes_merge_into_single_unified_polygon_without_overlap() {
        let mut map = SpatialMeshMap::default();
        let p1 = RawArPlane {
            raw_type: 0,
            pos: Vec3::new(0.0, -1.2, -1.5),
            rot: Quat::IDENTITY,
            extent_x: 1.2,
            extent_z: 1.2,
            polygon_xz: vec![[-0.6, -0.6], [0.6, -0.6], [0.6, 0.6], [-0.6, 0.6]],
        };
        let p2 = RawArPlane {
            raw_type: 0,
            pos: Vec3::new(0.5, -1.18, -1.5),
            rot: Quat::IDENTITY,
            extent_x: 1.0,
            extent_z: 1.2,
            polygon_xz: vec![[-0.5, -0.6], [0.5, -0.6], [0.5, 0.6], [-0.5, 0.6]],
        };
        map.ingest_ar_plane_polygons(&[p1, p2], 0.0);
        assert_eq!(map.canonical_plane_count(), 1);
        assert!(map.planes[0].extent_x >= 1.55);
        assert_eq!(map.planes[0].kind, SurfaceKind::Floor);

        let mut verts = Vec::new();
        let (_, _, plane_tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert!(plane_tris >= 8);
        assert_eq!(verts.len() % 3, 0);
    }

    #[test]
    fn tsdf_hugs_curved_object_with_outward_ccw_winding_and_diff_maintain_skips_unchanged() {
        let mut map = SpatialMeshMap::default();
        let dw = 80usize;
        let dh = 60usize;
        let fx = 64.0f32;
        let fy = 64.0f32;
        let cx = 40.0f32;
        let cy = 30.0f32;

        // 0. Low-confidence (conf = 20) hallucinated mid-air depth must produce ZERO triangles!
        let low_conf = vec![20u8; dw * dh];
        let depth_ghost = make_synthetic_depth_image(dw, dh, |_, _| 750u16);
        for _ in 0..5 {
            map.ingest_depth_map(
                &depth_ghost,
                dw,
                dh,
                dw * 2,
                2,
                Some(&low_conf),
                dw,
                dh,
                dw,
                1,
                false,
                fx,
                fy,
                cx,
                cy,
                dw,
                dh,
                Vec3::ZERO,
                Quat::IDENTITY,
                true,
                1,
            );
        }
        map.step_greedy_meshing(16);
        assert_eq!(map.confirmed_voxel_count(), 0, "low-confidence air ghost must never lock!");

        // Synthetic curved cylinder/mug at z = -0.90m..-1.02m centered in view, background wall at z = -2.2m
        let high_conf = vec![190u8; dw * dh];
        let depth_obj = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx.abs() < 16.0 && dy.abs() < 14.0 {
                let curve_mm = 900.0 + (dx * dx) * 0.45;
                curve_mm as u16
            } else {
                2200u16
            }
        });

        // Turbo scan for 3 independent frames locks the curved static object cleanly (weight >= 42)
        for _ in 0..3 {
            let st = map.ingest_depth_map(
                &depth_obj,
                dw,
                dh,
                dw * 2,
                2,
                Some(&high_conf),
                dw,
                dh,
                dw,
                1,
                false,
                fx,
                fy,
                cx,
                cy,
                dw,
                dh,
                Vec3::ZERO,
                Quat::IDENTITY,
                true,
                1,
            );
            assert!(st.valid_rays > 500);
        }
        assert!(map.has_dense_depth);
        let rebuilt = map.step_greedy_meshing(32);
        assert!(rebuilt >= 1);

        let mut verts = Vec::new();
        let (chunks, tris, _plane_tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert!(chunks >= 1);
        assert!(tris >= 40, "expected dense conformal triangles, got {}", tris);

        // Verify ALL front-facing triangles have outward CCW winding pointing back toward camera (+Z)
        let mut outward_ccw = 0usize;
        let mut near_curved_obj = 0usize;
        for tri in verts.chunks_exact(3) {
            let p0 = Vec3::from_array(tri[0].pos);
            let p1 = Vec3::from_array(tri[1].pos);
            let p2 = Vec3::from_array(tri[2].pos);
            let n = Vec3::from_array(tri[0].normal);
            let geo_n = (p1 - p0).cross(p2 - p0);
            if geo_n.length_squared() > 1e-8 && geo_n.dot(n) > 0.0 {
                outward_ccw += 1;
            }
            let mid = (p0 + p1 + p2) / 3.0;
            if mid.z > -1.15 && mid.z < -0.82 && mid.x.abs() < 0.30 {
                near_curved_obj += 1;
            }
        }
        assert!(
            outward_ccw * 10 >= (tris as usize) * 9,
            "outward CCW winding ratio too low: {}/{}",
            outward_ccw,
            tris
        );
        assert!(
            near_curved_obj >= 15,
            "mesh must hug curved foreground object at z~-0.95m, found {} tris",
            near_curved_obj
        );

        // Now LOCK (turbo = false) -> Compare & Maintain mode:
        // (a) Unchanged scene with +6cm realistic monocular depth wobble MUST update EXACTLY 0 voxels!
        let depth_wobble = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx.abs() < 16.0 && dy.abs() < 14.0 {
                (960.0 + (dx * dx) * 0.45) as u16
            } else {
                2260u16
            }
        });
        let maintain_wobble = map.ingest_depth_map(
            &depth_wobble,
            dw,
            dh,
            dw * 2,
            2,
            Some(&high_conf),
            dw,
            dh,
            dw,
            1,
            false,
            fx,
            fy,
            cx,
            cy,
            dw,
            dh,
            Vec3::ZERO,
            Quat::IDENTITY,
            false,
            2,
        );
        assert_eq!(
            maintain_wobble.updated_voxels, 0,
            "TRUE LOCK must ignore ±6cm depth wobble on unchanged room!"
        );

        // (b) Turning camera to an UNSCANNED direction (+90 deg) while LOCKED must update EXACTLY 0 voxels!
        let maintain_unscanned = map.ingest_depth_map(
            &depth_obj,
            dw,
            dh,
            dw * 2,
            2,
            Some(&high_conf),
            dw,
            dh,
            dw,
            1,
            false,
            fx,
            fy,
            cx,
            cy,
            dw,
            dh,
            Vec3::ZERO,
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            false,
            2,
        );
        assert_eq!(
            maintain_unscanned.updated_voxels, 0,
            "TRUE LOCK must NEVER scan brand-new unscanned directions!"
        );

        // (c) Single-frame removal glitch must NOT touch locked object (0 updates on tick 1 & 2)!
        let depth_empty = make_synthetic_depth_image(dw, dh, |_, _| 2200u16);
        let glitch_1 = map.ingest_depth_map(
            &depth_empty,
            dw,
            dh,
            dw * 2,
            2,
            Some(&high_conf),
            dw,
            dh,
            dw,
            1,
            false,
            fx,
            fy,
            cx,
            cy,
            dw,
            dh,
            Vec3::ZERO,
            Quat::IDENTITY,
            false,
            2,
        );
        assert_eq!(glitch_1.updated_voxels, 0, "1-frame glitch must not erode locked mesh");

        // (d) Persistent removal across 3+ consecutive ticks MUST confirm & carve away the removed object!
        let mut total_carve_updates = 0u32;
        for _ in 0..10 {
            let st = map.ingest_depth_map(
                &depth_empty,
                dw,
                dh,
                dw * 2,
                2,
                Some(&high_conf),
                dw,
                dh,
                dw,
                1,
                false,
                fx,
                fy,
                cx,
                cy,
                dw,
                dh,
                Vec3::ZERO,
                Quat::IDENTITY,
                false,
                2,
            );
            total_carve_updates += st.updated_voxels;
        }
        assert!(
            total_carve_updates > 20,
            "persistent removal must carve removed object, updated={}",
            total_carve_updates
        );
        map.step_greedy_meshing(32);
        let mut verts_after = Vec::new();
        map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts_after);
        let remaining_fg = verts_after
            .chunks_exact(3)
            .filter(|tri| {
                let mid = (Vec3::from_array(tri[0].pos)
                    + Vec3::from_array(tri[1].pos)
                    + Vec3::from_array(tri[2].pos))
                    / 3.0;
                mid.z > -1.25 && mid.x.abs() < 0.25 && mid.y.abs() < 0.22
            })
            .count();
        assert!(
            remaining_fg < near_curved_obj / 3,
            "removed object should be carved away: before={}, after={}",
            near_curved_obj,
            remaining_fg
        );
    }

    #[test]
    fn two_tier_separates_flat_wall_from_detail_object_and_smooth_anchored_interior_locks() {
        let mut map = SpatialMeshMap::default();
        // Register a canonical flat Wall plane at z = -2.0m (facing +Z toward camera)
        let wall_rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        map.ingest_ar_plane_polygons(
            &[RawArPlane {
                raw_type: 1,
                pos: Vec3::new(0.0, 0.0, -2.0),
                rot: wall_rot,
                extent_x: 3.0,
                extent_z: 2.4,
                polygon_xz: vec![[-1.5, -1.2], [1.5, -1.2], [1.5, 1.2], [-1.5, 1.2]],
            }],
            0.0,
        );
        assert_eq!(map.canonical_plane_count(), 1);

        let dw = 80usize;
        let dh = 60usize;
        let fx = 64.0f32;
        let fy = 64.0f32;
        let cx = 40.0f32;
        let cy = 30.0f32;

        // Protruding curved mug at z ~ -0.95m in front of the flat wall at z = -2.0m.
        // Simulate realistic sparse ARCore RawDepthConfidence: high conf (160) on mug border, 0 on smooth mug center!
        let depth = make_synthetic_depth_image(dw, dh, |x, y| {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            if dx.abs() < 12.0 && dy.abs() < 12.0 {
                (920.0 + (dx * dx) * 0.40) as u16
            } else {
                2000u16
            }
        });
        let mut sparse_conf = vec![160u8; dw * dh];
        for y in 0..dh {
            for x in 0..dw {
                let dx = (x as f32 - cx).abs();
                let dy = (y as f32 - cy).abs();
                // Smooth inner core of mug has conf = 0, but is anchored within ±4px to confident border!
                if dx < 4.0 && dy < 4.0 {
                    sparse_conf[y * dw + x] = 0;
                }
            }
        }

        for _ in 0..4 {
            map.ingest_depth_map(
                &depth,
                dw,
                dh,
                dw * 2,
                2,
                Some(&sparse_conf),
                dw,
                dh,
                dw,
                1,
                false,
                fx,
                fy,
                cx,
                cy,
                dw,
                dh,
                Vec3::ZERO,
                Quat::IDENTITY,
                true,
                1,
            );
        }
        map.step_greedy_meshing(32);

        let mut verts = Vec::new();
        let (_, detail_tris, plane_tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert!(detail_tris >= 24, "detail object must build crisp TSDF tris, got {}", detail_tris);
        assert!(plane_tris >= 8, "flat wall must stay sleek Tier-1 architectural plane, got {}", plane_tris);

        // Almost ALL Tier-2 TSDF triangles should belong to the protruding mug (z > -1.3m), NOT the flat wall at z = -2.0m!
        let detail_slice = &verts[..(detail_tris as usize) * 3];
        let wall_micro_tris = detail_slice
            .chunks_exact(3)
            .filter(|tri| (tri[0].pos[2] + tri[1].pos[2] + tri[2].pos[2]) / 3.0 < -1.75)
            .count();
        assert!(
            wall_micro_tris * 5 < detail_tris as usize,
            "flat wall should not waste TSDF detail tris: wall_micro={}, detail={}",
            wall_micro_tris,
            detail_tris
        );
    }
}
