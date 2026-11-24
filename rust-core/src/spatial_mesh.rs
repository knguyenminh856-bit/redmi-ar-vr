use crate::passthrough::compile_program;
use crate::room_scanner::{
    classify_ar_plane, infer_floor_y, AabbCollider, PlaneMarker, SurfaceKind,
};
use glam::{Mat3, Mat4, Quat, Vec3};
use glow::HasContext;
use std::collections::HashMap;
use std::mem::size_of;

pub const CHUNK_DIM: usize = 16;
pub const CHUNK_VOLUME: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM; // 4096 bytes = 4 KB/chunk
pub const CELL_SIZE: f32 = 0.06; // 6cm voxels -> 0.96m cube per chunk
pub const MAX_ACTIVE_CHUNKS: usize = 128; // Hard RAM cap (~512 KB raw occupancy)
pub const MAX_CANONICAL_PLANES: usize = 32;
pub const MAX_POLY_VERTS: usize = 24;
pub const MAX_MESH_VERTICES: usize = 48_000; // Hard Mali-G57 GPU cap (16,000 triangles)
pub const MAX_RENDER_RADIUS_M: f32 = 6.0;

pub const SOLID_THRESHOLD: u8 = 48; // Confirmed solid after multi-view accumulation
pub const CARVE_DROP_THRESHOLD: u8 = 24; // Hysteresis prevents surface flickering

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
    vec3 a3 = smoothstep(vec3(0.0), max(d * 1.45, vec3(0.002)), v_bary);
    float edge_factor = 1.0 - min(min(a3.x, a3.y), a3.z);

    vec3 n = normalize(v_normal);
    float shade = 0.62 + 0.38 * abs(dot(n, normalize(vec3(0.35, 0.85, 0.45))));
    vec3 fill_rgb = v_color.rgb * shade;
    vec3 wire_rgb = mix(v_color.rgb, vec3(1.0), 0.38);

    float fill_a = v_color.a;
    float wire_a = clamp(v_color.a * 2.4 + 0.42, 0.50, 0.86);

    vec3 out_rgb = mix(fill_rgb, wire_rgb, edge_factor);
    float out_a = mix(fill_a, wire_a, edge_factor);
    frag_color = vec4(out_rgb, out_a);
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

struct VoxelChunk {
    cells: Box<[u8; CHUNK_VOLUME]>,
    solid_mask: Box<[bool; CHUNK_VOLUME]>,
    solid_count: u16,
    dirty: bool,
    last_touched_gen: u64,
    tris: Vec<TriMeshVertex>,
    aabbs: Vec<AabbCollider>,
}

impl VoxelChunk {
    fn new(gen: u64) -> Self {
        Self {
            cells: Box::new([0u8; CHUNK_VOLUME]),
            solid_mask: Box::new([false; CHUNK_VOLUME]),
            solid_count: 0,
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
}

pub struct SpatialMeshMap {
    pub planes: Vec<CanonicalPlane>,
    chunks: HashMap<(i32, i32, i32), VoxelChunk>,
    pub tick_gen: u64,
    pub mesh_gen: u64,
}

impl Default for SpatialMeshMap {
    fn default() -> Self {
        Self {
            planes: Vec::with_capacity(MAX_CANONICAL_PLANES),
            chunks: HashMap::with_capacity(64),
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

impl SpatialMeshMap {
    pub fn clear(&mut self) {
        self.planes.clear();
        self.chunks.clear();
        self.mesh_gen = self.mesh_gen.wrapping_add(1);
    }

    pub fn active_chunk_count(&self) -> usize {
        self.chunks.values().filter(|c| c.solid_count > 0).count()
    }

    pub fn confirmed_voxel_count(&self) -> usize {
        self.chunks.values().map(|c| c.solid_count as usize).sum()
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

    /// Rule 1: Coplanar Plane Merging — fuses incoming ARCore true polygons into single canonical planes with ZERO overlap.
    pub fn ingest_ar_plane_polygons(&mut self, incoming: &[RawArPlane], cam_y: f32) {
        self.tick_gen = self.tick_gen.wrapping_add(1);
        if incoming.is_empty() {
            return;
        }

        let mut summary: Vec<(i32, Vec3, f32, f32)> = Vec::with_capacity(self.planes.len() + incoming.len());
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

            // Search for coplanar overlapping/adjacent canonical plane
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
                });
                changed = true;
            }
        }

        if changed {
            self.coalesce_overlapping_planes(cam_y, floor_y);
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

        // Weighted height/position fusion (area-weighted EMA prevents jumping)
        let w_new = new_area.clamp(0.12, 4.0);
        let alpha = (w_new / (old.weight + w_new)).clamp(0.12, 0.45);
        let plane_anchor = old.pos + old.normal * ((new_pos - old.pos).dot(old.normal) * alpha);

        // Collect all polygon points from old + new projected onto the canonical plane frame
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
    fn split_cell(gx: i32, gy: i32, gz: i32) -> ((i32, i32, i32), usize, usize, usize) {
        let d = CHUNK_DIM as i32;
        (
            (gx.div_euclid(d), gy.div_euclid(d), gz.div_euclid(d)),
            gx.rem_euclid(d) as usize,
            gy.rem_euclid(d) as usize,
            gz.rem_euclid(d) as usize,
        )
    }

    fn is_solid_global(&self, gx: i32, gy: i32, gz: i32) -> bool {
        let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
        self.chunks
            .get(&ckey)
            .map(|c| c.solid_mask[VoxelChunk::idx(lx, ly, lz)])
            .unwrap_or(false)
    }

    fn point_flush_on_canonical_plane(&self, pt: Vec3) -> bool {
        self.planes.iter().any(|p| {
            let rel = pt - p.pos;
            if rel.dot(p.normal).abs() >= 0.055 {
                return false;
            }
            let lx = rel.dot(p.rot * Vec3::X).abs();
            let lz = rel.dot(p.rot * Vec3::Z).abs();
            lx <= p.extent_x * 0.55 + 0.08 && lz <= p.extent_z * 0.55 + 0.08
        })
    }

    /// Rule 1 & 2: Multi-view Occupancy Accumulation + Free-Space Ray Carving with LRU Chunk Cap.
    pub fn ingest_point_cloud(&mut self, point_count: usize, packed_xyzc: &[f32], cam_pos: Vec3) {
        self.tick_gen = self.tick_gen.wrapping_add(1);
        let gen = self.tick_gen;
        let n = point_count.min(packed_xyzc.len() / 4);

        for i in 0..n {
            let base = i * 4;
            let conf = packed_xyzc[base + 3];
            if !conf.is_finite() || conf < 0.38 {
                continue;
            }
            let pt = Vec3::new(packed_xyzc[base], packed_xyzc[base + 1], packed_xyzc[base + 2]);
            if !pt.is_finite() {
                continue;
            }
            let delta = pt - cam_pos;
            let dist_sq = delta.length_squared();
            if !(0.25 * 0.25..=3.5 * 3.5).contains(&dist_sq) {
                continue;
            }
            let dist = dist_sq.sqrt();

            // Free-Space Carving along camera ray every 3rd point so moved furniture erases itself cleanly
            // Step 0.055m (< 0.06m cell size) guarantees no 6cm voxel is skipped along the ray.
            if i % 3 == 0 && dist > 0.40 {
                let dir = delta / dist;
                let max_t = (dist - 0.12).min(2.2);
                let mut t = 0.24f32;
                let mut steps = 0usize;
                while t < max_t && steps < 36 {
                    let sample = cam_pos + dir * t;
                    let (gx, gy, gz) = Self::world_to_cell(sample);
                    let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
                    if let Some(chunk) = self.chunks.get_mut(&ckey) {
                        let idx = VoxelChunk::idx(lx, ly, lz);
                        if chunk.cells[idx] > 0 {
                            chunk.cells[idx] = chunk.cells[idx].saturating_sub(14);
                            if chunk.solid_mask[idx] && chunk.cells[idx] < CARVE_DROP_THRESHOLD {
                                chunk.solid_mask[idx] = false;
                                chunk.solid_count = chunk.solid_count.saturating_sub(1);
                                chunk.dirty = true;
                            }
                            chunk.last_touched_gen = gen;
                        }
                    }
                    t += 0.055;
                    steps += 1;
                }
            }

            // Skip points lying flat on already-reconstructed planes to avoid bumpy voxel clutter
            if self.point_flush_on_canonical_plane(pt) {
                continue;
            }

            let (gx, gy, gz) = Self::world_to_cell(pt);
            let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
            if !self.chunks.contains_key(&ckey) {
                if self.chunks.len() >= MAX_ACTIVE_CHUNKS {
                    if let Some(&evict_key) = self
                        .chunks
                        .iter()
                        .min_by_key(|(_, c)| (c.solid_count > 0, c.last_touched_gen))
                        .map(|(k, _)| k)
                    {
                        self.chunks.remove(&evict_key);
                    }
                }
                self.chunks.insert(ckey, VoxelChunk::new(gen));
            }
            if let Some(chunk) = self.chunks.get_mut(&ckey) {
                let idx = VoxelChunk::idx(lx, ly, lz);
                chunk.cells[idx] = chunk.cells[idx].saturating_add(24);
                chunk.last_touched_gen = gen;
                if !chunk.solid_mask[idx] && chunk.cells[idx] >= SOLID_THRESHOLD {
                    chunk.solid_mask[idx] = true;
                    chunk.solid_count = chunk.solid_count.saturating_add(1);
                    chunk.dirty = true;
                }
            }
        }

        // Prune completely empty stale chunks to keep memory tight
        self.chunks
            .retain(|_, c| c.solid_count > 0 || gen.saturating_sub(c.last_touched_gen) <= 60);
    }

    /// Rule 3: Budgeted Incremental Greedy Meshing — rebuilds at most `max_dirty_chunks` per frame.
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

        for ckey in &dirty_keys {
            let (tris, aabbs) = self.build_chunk_greedy_geometry(*ckey);
            if let Some(chunk) = self.chunks.get_mut(ckey) {
                chunk.tris = tris;
                chunk.aabbs = aabbs;
                chunk.dirty = false;
            }
        }

        self.mesh_gen = self.mesh_gen.wrapping_add(1);
        dirty_keys.len()
    }

    fn build_chunk_greedy_geometry(
        &self,
        ckey: (i32, i32, i32),
    ) -> (Vec<TriMeshVertex>, Vec<AabbCollider>) {
        let Some(chunk) = self.chunks.get(&ckey) else {
            return (Vec::new(), Vec::new());
        };
        if chunk.solid_count == 0 {
            return (Vec::new(), Vec::new());
        }

        let base_gx = ckey.0 * (CHUNK_DIM as i32);
        let base_gy = ckey.1 * (CHUNK_DIM as i32);
        let base_gz = ckey.2 * (CHUNK_DIM as i32);
        let mut tris = Vec::with_capacity((chunk.solid_count as usize) * 6);
        let col = [0.18, 0.92, 0.72, 0.22];

        // Face-culled 2D greedy rectangle merging across all 6 cardinal directions
        let dirs: [(usize, i32, Vec3); 6] = [
            (0, 1, Vec3::X),
            (0, -1, Vec3::NEG_X),
            (1, 1, Vec3::Y),
            (1, -1, Vec3::NEG_Y),
            (2, 1, Vec3::Z),
            (2, -1, Vec3::NEG_Z),
        ];

        for (axis, sign, normal) in dirs {
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;

            for d in 0..CHUNK_DIM {
                let mut mask = [[false; CHUNK_DIM]; CHUNK_DIM];
                let mut any_face = false;

                for v in 0..CHUNK_DIM {
                    for u in 0..CHUNK_DIM {
                        let mut c = [0usize; 3];
                        c[axis] = d;
                        c[u_axis] = u;
                        c[v_axis] = v;
                        if !chunk.solid_mask[VoxelChunk::idx(c[0], c[1], c[2])] {
                            continue;
                        }
                        let mut nb = [base_gx + c[0] as i32, base_gy + c[1] as i32, base_gz + c[2] as i32];
                        nb[axis] += sign;
                        if !self.is_solid_global(nb[0], nb[1], nb[2]) {
                            mask[v][u] = true;
                            any_face = true;
                        }
                    }
                }

                if !any_face {
                    continue;
                }

                for v in 0..CHUNK_DIM {
                    let mut u = 0;
                    while u < CHUNK_DIM {
                        if !mask[v][u] {
                            u += 1;
                            continue;
                        }
                        let mut w = 1;
                        while u + w < CHUNK_DIM && mask[v][u + w] {
                            w += 1;
                        }
                        let mut h = 1;
                        'grow_h: while v + h < CHUNK_DIM {
                            for k in 0..w {
                                if !mask[v + h][u + k] {
                                    break 'grow_h;
                                }
                            }
                            h += 1;
                        }
                        for dv in 0..h {
                            for du in 0..w {
                                mask[v + dv][u + du] = false;
                            }
                        }

                        // Emit 1 merged quad (2 triangles) on face plane `d + (sign > 0 ? 1 : 0)`
                        let plane_d = (d as i32 + if sign > 0 { 1 } else { 0 }) as f32;
                        let mut o = [base_gx as f32, base_gy as f32, base_gz as f32];
                        o[axis] += plane_d;
                        o[u_axis] += u as f32;
                        o[v_axis] += v as f32;

                        let mut du_vec = [0.0f32; 3];
                        let mut dv_vec = [0.0f32; 3];
                        du_vec[u_axis] = w as f32 * CELL_SIZE;
                        dv_vec[v_axis] = h as f32 * CELL_SIZE;

                        let p0 = Vec3::new(o[0] * CELL_SIZE, o[1] * CELL_SIZE, o[2] * CELL_SIZE);
                        let du_v = Vec3::from_array(du_vec);
                        let dv_v = Vec3::from_array(dv_vec);
                        if sign > 0 {
                            push_mesh_quad(&mut tris, p0, p0 + du_v, p0 + du_v + dv_v, p0 + dv_v, normal, col);
                        } else {
                            push_mesh_quad(&mut tris, p0, p0 + dv_v, p0 + du_v + dv_v, p0 + du_v, normal, col);
                        }
                        u += w;
                    }
                }
            }
        }

        // Extract merged per-layer physics AABBs (reduces 64 tiny voxel boxes down to compact merged slabs)
        let mut aabbs = Vec::with_capacity(16);
        for ly in 0..CHUNK_DIM {
            let mut mask = [[false; CHUNK_DIM]; CHUNK_DIM];
            let mut any = false;
            for lz in 0..CHUNK_DIM {
                for lx in 0..CHUNK_DIM {
                    if chunk.solid_mask[VoxelChunk::idx(lx, ly, lz)] {
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

        (tris, aabbs)
    }

    /// Exports canonical planes for physics and clean mode rendering.
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

    /// Collects nearby merged chunk AABBs for physics collisions (within 3.0m, capped at 96 slabs).
    pub fn collect_physics_chunk_aabbs(&self, cam_pos: Vec3) -> Vec<AabbCollider> {
        let mut nearby: Vec<(f32, &VoxelChunk)> = self
            .chunks
            .iter()
            .filter_map(|(&ckey, chunk)| {
                if chunk.aabbs.is_empty() {
                    return None;
                }
                let center = Vec3::new(
                    (ckey.0 as f32 + 0.5) * (CHUNK_DIM as f32 * CELL_SIZE),
                    (ckey.1 as f32 + 0.5) * (CHUNK_DIM as f32 * CELL_SIZE),
                    (ckey.2 as f32 + 0.5) * (CHUNK_DIM as f32 * CELL_SIZE),
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

        let mut out = Vec::with_capacity(64);
        for (_, chunk) in nearby {
            for &b in &chunk.aabbs {
                if out.len() >= 96 {
                    return out;
                }
                out.push(b);
            }
        }
        out
    }

    /// Builds exact visible Triangle Mesh sheets (triangulated canonical polygons + Greedy-Meshed chunks)
    /// respecting Frustum + 6m Radius Culling and the `MAX_MESH_VERTICES` GPU cap.
    pub fn build_triangle_mesh_buffer(
        &self,
        cam_pos: Vec3,
        cam_fwd: Vec3,
        out: &mut Vec<TriMeshVertex>,
    ) -> (u32, u32) {
        out.clear();

        // 1. Triangulate every canonical merged ARCore plane polygon
        for plane in &self.planes {
            if out.len() + 200 > MAX_MESH_VERTICES {
                break;
            }
            if (plane.pos - cam_pos).length() > MAX_RENDER_RADIUS_M + plane.bounding_radius() {
                continue;
            }
            triangulate_canonical_plane(plane, out);
        }

        // 2. Append Greedy-Meshed 3D obstacle surface triangles with distance + forward frustum culling
        let chunk_span = CHUNK_DIM as f32 * CELL_SIZE;
        let mut active_chunks = 0u32;
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
            if dist > 1.2 && to_c.dot(cam_fwd) < -1.0 {
                continue;
            }
            active_chunks += 1;
            let remain = MAX_MESH_VERTICES.saturating_sub(out.len());
            let copy_n = (chunk.tris.len().min(remain) / 3) * 3;
            if copy_n > 0 {
                out.extend_from_slice(&chunk.tris[..copy_n]);
            }
        }

        (active_chunks, (out.len() / 3) as u32)
    }

    /// Exports all solid voxels grouped by chunk coordinate as compact `(local_idx, weight)` pairs.
    pub fn export_solid_chunks(&self) -> Vec<((i32, i32, i32), Vec<(u16, u8)>)> {
        let mut out = Vec::new();
        for (&ckey, chunk) in &self.chunks {
            if chunk.solid_count == 0 {
                continue;
            }
            let mut entries = Vec::with_capacity(chunk.solid_count as usize);
            for idx in 0..CHUNK_VOLUME {
                if chunk.solid_mask[idx] {
                    entries.push((idx as u16, chunk.cells[idx]));
                }
            }
            if !entries.is_empty() {
                out.push((ckey, entries));
            }
        }
        out
    }

    /// Transforms saved planes and voxels by a 4-DoF alignment `(q_align, t_align)` and merges them
    /// cleanly into the live canonical map so old geometry and new real-time observations coexist.
    pub fn import_and_merge_transformed(
        &mut self,
        saved_planes: &[CanonicalPlane],
        saved_voxels_world: &[(Vec3, u8)],
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
        for &(w_saved, weight) in saved_voxels_world {
            let w_curr = q_align * w_saved + t_align;
            if !w_curr.is_finite() || self.point_flush_on_canonical_plane(w_curr) {
                continue;
            }
            let (gx, gy, gz) = Self::world_to_cell(w_curr);
            let (ckey, lx, ly, lz) = Self::split_cell(gx, gy, gz);
            if !self.chunks.contains_key(&ckey) {
                if self.chunks.len() >= MAX_ACTIVE_CHUNKS {
                    continue;
                }
                self.chunks.insert(ckey, VoxelChunk::new(gen));
            }
            if let Some(chunk) = self.chunks.get_mut(&ckey) {
                let idx = VoxelChunk::idx(lx, ly, lz);
                // Cap restored prior weight at 96 so Free-Space Carving can cleanly erase moved furniture
                let prior = weight.clamp(SOLID_THRESHOLD + 12, 96);
                if chunk.cells[idx] < prior {
                    chunk.cells[idx] = prior;
                }
                chunk.last_touched_gen = gen;
                if !chunk.solid_mask[idx] && chunk.cells[idx] >= SOLID_THRESHOLD {
                    chunk.solid_mask[idx] = true;
                    chunk.solid_count = chunk.solid_count.saturating_add(1);
                    chunk.dirty = true;
                }
            }
        }
        // Rebuild up to 32 restored chunks immediately so the restored room is visible right away
        self.step_greedy_meshing(32);
        self.mesh_gen = self.mesh_gen.wrapping_add(1);
    }

    /// Exports all reconstructed plane and obstacle triangles grouped by semantic label for `.obj` export.
    pub fn build_obj_mesh_groups(&mut self) -> Vec<(&'static str, Vec<TriMeshVertex>)> {
        self.step_greedy_meshing(MAX_ACTIVE_CHUNKS);
        let mut groups: Vec<(&'static str, Vec<TriMeshVertex>)> = vec![
            ("Floor", Vec::new()),
            ("Table", Vec::new()),
            ("Seat", Vec::new()),
            ("Wall", Vec::new()),
            ("Ceiling", Vec::new()),
            ("Obstacles", Vec::new()),
        ];
        for plane in &self.planes {
            let slot = match plane.kind {
                SurfaceKind::Floor => 0,
                SurfaceKind::Table => 1,
                SurfaceKind::Seat => 2,
                SurfaceKind::Wall | SurfaceKind::VerticalObject => 3,
                SurfaceKind::Ceiling => 4,
                SurfaceKind::Unknown => continue,
            };
            triangulate_canonical_plane(plane, &mut groups[slot].1);
        }
        for chunk in self.chunks.values() {
            if !chunk.tris.is_empty() {
                groups[5].1.extend_from_slice(&chunk.tris);
            }
        }
        groups.into_iter().filter(|(_, v)| !v.is_empty()).collect()
    }
}

/// Triangulates a convex canonical plane polygon with a 2-ring subdivision for large surfaces so
/// triangle sheets remain crisp, uniform, and visually inspectable.
pub fn triangulate_canonical_plane(plane: &CanonicalPlane, out: &mut Vec<TriMeshVertex>) {
    let n = plane.polygon_xz.len();
    if n < 3 {
        return;
    }
    let ax_x = plane.rot * Vec3::X;
    let ax_z = plane.rot * Vec3::Z;
    let normal = plane.normal;
    let origin = plane.pos + normal * 0.003;
    let rgb = surface_rgb(plane.kind);
    let col = [rgb[0], rgb[1], rgb[2], 0.16];

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

    let to_world = |x: f32, z: f32| origin + ax_x * x + ax_z * z;
    let center_w = to_world(cx, cz);

    if avg_r > 0.42 {
        // Subdivided 2-ring mesh (inner ring at 50% radius -> crisp uniform triangles)
        for i in 0..n {
            let [x0, z0] = plane.polygon_xz[i];
            let [x1, z1] = plane.polygon_xz[(i + 1) % n];
            let m0 = to_world(cx + (x0 - cx) * 0.5, cz + (z0 - cz) * 0.5);
            let m1 = to_world(cx + (x1 - cx) * 0.5, cz + (z1 - cz) * 0.5);
            let p0 = to_world(x0, z0);
            let p1 = to_world(x1, z1);

            push_mesh_tri(out, center_w, m0, m1, normal, col);
            push_mesh_quad(out, m0, p0, p1, m1, normal, col);
        }
    } else {
        for i in 0..n {
            let [x0, z0] = plane.polygon_xz[i];
            let [x1, z1] = plane.polygon_xz[(i + 1) % n];
            push_mesh_tri(out, center_w, to_world(x0, z0), to_world(x1, z1), normal, col);
        }
    }
}

pub struct TriMeshRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    uploaded_gen: u64,
    vert_count: i32,
}

impl TriMeshRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, TRI_MESH_VERT, TRI_MESH_FRAG)?;
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                (MAX_MESH_VERTICES * size_of::<TriMeshVertex>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<TriMeshVertex>() as i32;
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

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            Ok(Self {
                program,
                vao,
                vbo,
                u_mvp,
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

    pub fn draw(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.vert_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.vert_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.vbo);
            gl.delete_vertex_array(self.vao);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Must fuse into 1 single canonical plane spanning X ~1.6m (NOT 2 overlapping rectangles)
        assert_eq!(map.canonical_plane_count(), 1);
        assert!(map.planes[0].extent_x >= 1.55);
        assert_eq!(map.planes[0].kind, SurfaceKind::Floor);

        let mut verts = Vec::new();
        let (_, tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert!(tris >= 4);
        assert_eq!(verts.len() % 3, 0);
        // Every triangle has proper unit barycentric coordinates
        assert_eq!(verts[0].bary, [1.0, 0.0, 0.0]);
        assert_eq!(verts[1].bary, [0.0, 1.0, 0.0]);
        assert_eq!(verts[2].bary, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn greedy_mesher_merges_adjacent_voxels_and_free_space_carving_erases_moved_objects() {
        let mut map = SpatialMeshMap::default();
        // Build a 2x2x1 solid block of 4 adjacent 6cm cells at z = -1.0m (3 hits each -> solid)
        let mut pts = Vec::new();
        for _ in 0..3 {
            for ix in 0..2 {
                for iy in 0..2 {
                    pts.extend_from_slice(&[
                        0.03 + ix as f32 * CELL_SIZE,
                        0.03 + iy as f32 * CELL_SIZE,
                        -1.02,
                        0.90,
                    ]);
                }
            }
        }
        map.ingest_point_cloud(pts.len() / 4, &pts, Vec3::ZERO);
        assert_eq!(map.confirmed_voxel_count(), 4);
        let rebuilt = map.step_greedy_meshing(4);
        assert_eq!(rebuilt, 1);

        let mut verts = Vec::new();
        let (chunks, tris) = map.build_triangle_mesh_buffer(Vec3::ZERO, Vec3::NEG_Z, &mut verts);
        assert_eq!(chunks, 1);
        // A 2x2x1 rectangular block greedy-meshes to EXACTLY 6 merged faces = 12 triangles (instead of 4 * 12 = 48!)
        assert_eq!(tris, 12);

        // Now simulate moving the object away: rays from camera hit a background wall at z = -2.2m
        // passing straight through the old block at z = -1.02m -> Free-Space Carving must erase it!
        let mut carve_pts = Vec::new();
        for ix in 0..2 {
            for iy in 0..2 {
                let target_dir = Vec3::new(
                    0.03 + ix as f32 * CELL_SIZE,
                    0.03 + iy as f32 * CELL_SIZE,
                    -1.02,
                )
                .normalize();
                let far = target_dir * 2.2;
                // Place at indices divisible by 3 so carving fires on every target ray
                carve_pts.extend_from_slice(&[far.x, far.y, far.z, 0.90]);
                carve_pts.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
                carve_pts.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            }
        }
        for _ in 0..8 {
            map.ingest_point_cloud(carve_pts.len() / 4, &carve_pts, Vec3::ZERO);
        }
        map.step_greedy_meshing(4);
        // Old foreground cells at z = -1.02m are carved away while new background wall cells at z = -2.2m exist
        for ix in 0..2 {
            for iy in 0..2 {
                let (gx, gy, gz) = SpatialMeshMap::world_to_cell(Vec3::new(
                    0.03 + ix as f32 * CELL_SIZE,
                    0.03 + iy as f32 * CELL_SIZE,
                    -1.02,
                ));
                assert!(!map.is_solid_global(gx, gy, gz));
            }
        }
        assert_eq!(map.confirmed_voxel_count(), 4);
    }
}
