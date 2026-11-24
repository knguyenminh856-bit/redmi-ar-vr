use crate::passthrough::compile_program;
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::collections::HashMap;
use std::mem::size_of;
use std::time::Instant;

pub const MAX_PLANES: usize = 12;
pub const FLOATS_PER_PLANE: usize = 10; // [raw_type, px, py, pz, qx, qy, qz, qw, extent_x, extent_z]

pub const MAX_OBJECTS: usize = 6;
pub const FLOATS_PER_OBJECT: usize = 6; // [kind_id, ndc_x, ndc_y, ndc_w, ndc_h, score]

pub const MAX_LABELS: usize = 6;
pub const FLOATS_PER_LABEL: usize = 6; // [kind_id, left_u, left_v, right_u, right_v, score]

pub const MAX_CACHED_PLANES: usize = 64;
pub const MAX_CACHED_OBJECTS: usize = 32;
pub const PLANE_CACHE_TTL_MS: u128 = 45_000;
pub const HIT_PATCH_TTL_MS: u128 = 20_000;
pub const OBJECT_CACHE_TTL_MS: u128 = 10_000;

pub const VOXEL_SIZE: f32 = 0.06;
pub const MAX_VOXELS: usize = 2048;
pub const VOXEL_TTL_MS: u128 = 35_000;
pub const VOXEL_NOISE_TTL_MS: u128 = 8_000;
pub const MAX_POINT_CLOUD_PTS: usize = 512;

const MAX_WORLD_VERTICES: usize = 16384;

const SCAN_VERT: &str = r#"
precision highp float;
attribute vec3 a_pos;
attribute vec4 a_color;
uniform mat4 u_mvp;
varying vec4 v_color;
void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const SCAN_FRAG: &str = r#"
precision mediump float;
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Unknown = 0,
    Floor = 1,
    Wall = 2,
    Table = 3,
    Seat = 4,
    Ceiling = 5,
    VerticalObject = 6,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ScanVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
pub struct PlaneMarker {
    pub kind: SurfaceKind,
    pub pos: Vec3,
    pub rot: Quat,
    pub extent_x: f32,
    pub extent_z: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct ObjectMarker {
    pub kind_id: i32,
    pub ndc_x: f32,
    pub ndc_y: f32,
    pub ndc_w: f32,
    pub ndc_h: f32,
    pub score: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct WorldObjectMarker {
    pub kind_id: i32,
    pub world_pos: Vec3,
    pub radius_hint_ndc: f32,
    pub score: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct AabbCollider {
    pub min: Vec3,
    pub max: Vec3,
    pub is_object_box: bool,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ProjectedLabel {
    pub kind_id: i32,
    pub left_u: f32,
    pub left_v: f32,
    pub right_u: f32,
    pub right_v: f32,
    pub score: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ReticleHit {
    pub valid: bool,
    pub hit_kind: i32, // 0=none, 1=Plane, 2=Point/Instant, 3=FloorFallback
    pub surface_kind: SurfaceKind,
    pub pos: Vec3,
    pub normal: Vec3,
    pub distance: f32,
}

impl Default for ReticleHit {
    fn default() -> Self {
        Self {
            valid: false,
            hit_kind: 0,
            surface_kind: SurfaceKind::Unknown,
            pos: Vec3::ZERO,
            normal: Vec3::Y,
            distance: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct CachedPlaneEntry {
    marker: PlaneMarker,
    normal: Vec3,
    last_seen: Instant,
    ttl_ms: u128,
    is_hit_patch: bool,
}

#[derive(Copy, Clone, Debug)]
struct CachedObjectEntry {
    marker: WorldObjectMarker,
    last_seen: Instant,
}

#[derive(Copy, Clone, Debug)]
struct VoxelCell {
    sum_pos: Vec3,
    hits: u32,
    last_seen: Instant,
}

pub struct SpatialMemoryCache {
    planes: Vec<CachedPlaneEntry>,
    objects: Vec<CachedObjectEntry>,
    voxels: HashMap<(i32, i32, i32), VoxelCell>,
}

impl Default for SpatialMemoryCache {
    fn default() -> Self {
        Self {
            planes: Vec::with_capacity(MAX_CACHED_PLANES),
            objects: Vec::with_capacity(MAX_CACHED_OBJECTS),
            voxels: HashMap::with_capacity(512),
        }
    }
}

impl SpatialMemoryCache {
    pub fn clear(&mut self) {
        self.planes.clear();
        self.objects.clear();
        self.voxels.clear();
    }

    pub fn clear_objects(&mut self) {
        self.objects.clear();
    }

    pub fn cached_plane_count(&self) -> usize {
        self.planes.len()
    }

    pub fn cached_object_count(&self) -> usize {
        self.objects.len()
    }

    pub fn confirmed_voxel_count(&self) -> usize {
        self.voxels.values().filter(|v| v.hits >= 3).count()
    }

    pub fn inferred_floor_y(&self) -> Option<f32> {
        self.planes
            .iter()
            .filter(|p| p.marker.kind == SurfaceKind::Floor)
            .map(|p| p.marker.pos.y)
            .reduce(f32::min)
    }

    pub fn ingest_ar_planes(&mut self, count: usize, packed: &[f32], cam_y: f32, scan_mode: i32) {
        self.ingest_ar_planes_at(count, packed, cam_y, scan_mode, Instant::now());
    }

    pub fn ingest_ar_planes_at(
        &mut self,
        count: usize,
        packed: &[f32],
        cam_y: f32,
        scan_mode: i32,
        now: Instant,
    ) {
        self.planes
            .retain(|p| now.duration_since(p.last_seen).as_millis() <= p.ttl_ms);

        if scan_mode <= 0 || count == 0 {
            return;
        }

        let n = count.min(MAX_PLANES);
        let mut raw_entries = Vec::with_capacity(n);
        let mut summary = Vec::with_capacity(n + self.planes.len());

        // Seed floor inference with previously remembered horizontal upward planes
        for existing in &self.planes {
            if matches!(
                existing.marker.kind,
                SurfaceKind::Floor | SurfaceKind::Seat | SurfaceKind::Table
            ) {
                summary.push((
                    0,
                    existing.marker.pos,
                    existing.marker.extent_x,
                    existing.marker.extent_z,
                ));
            }
        }

        for i in 0..n {
            let base = i * FLOATS_PER_PLANE;
            if base + FLOATS_PER_PLANE > packed.len() {
                break;
            }
            let raw_type = packed[base].round() as i32;
            let pos = Vec3::new(packed[base + 1], packed[base + 2], packed[base + 3]);
            let rot = Quat::from_xyzw(
                packed[base + 4],
                packed[base + 5],
                packed[base + 6],
                packed[base + 7],
            );
            let rot = if rot.length_squared() > 1e-6 {
                rot.normalize()
            } else {
                Quat::IDENTITY
            };
            let ex = packed[base + 8].clamp(0.05, 8.0);
            let ez = packed[base + 9].clamp(0.05, 8.0);
            if pos.is_finite() {
                summary.push((raw_type, pos, ex, ez));
                raw_entries.push((raw_type, pos, rot, ex, ez));
            }
        }

        let floor_y = infer_floor_y(&summary, cam_y);
        for (raw_type, pos, rot, ex, ez) in raw_entries {
            let kind = classify_ar_plane(raw_type, pos.y, ex, ez, cam_y, floor_y);
            if kind == SurfaceKind::Unknown {
                continue;
            }
            let normal = (rot * Vec3::Y).normalize_or_zero();
            if normal.length_squared() < 1e-4 {
                continue;
            }

            let existing_idx = self.planes.iter().position(|old| {
                let normal_dot = old.normal.dot(normal);
                let plane_off = (pos - old.marker.pos).dot(old.normal).abs();
                let center_dist = (pos - old.marker.pos).length();
                normal_dot > 0.86 && plane_off < 0.18 && center_dist < 0.85
            });

            if let Some(idx) = existing_idx {
                let old = &mut self.planes[idx];
                let aligned_rot = if old.marker.rot.dot(rot) < 0.0 {
                    -rot
                } else {
                    rot
                };
                old.marker.pos = old.marker.pos * 0.65 + pos * 0.35;
                old.marker.rot = old.marker.rot.slerp(aligned_rot, 0.35).normalize();
                old.marker.extent_x = old.marker.extent_x.max(ex);
                old.marker.extent_z = old.marker.extent_z.max(ez);
                old.marker.kind = classify_ar_plane(
                    raw_type,
                    old.marker.pos.y,
                    old.marker.extent_x,
                    old.marker.extent_z,
                    cam_y,
                    floor_y,
                );
                old.normal = (old.marker.rot * Vec3::Y).normalize_or_zero();
                old.last_seen = now;
                old.ttl_ms = PLANE_CACHE_TTL_MS;
                old.is_hit_patch = false;
            } else {
                if self.planes.len() >= MAX_CACHED_PLANES {
                    if let Some((oldest_idx, _)) = self
                        .planes
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_seen)
                    {
                        self.planes.swap_remove(oldest_idx);
                    }
                }
                self.planes.push(CachedPlaneEntry {
                    marker: PlaneMarker {
                        kind,
                        pos,
                        rot,
                        extent_x: ex,
                        extent_z: ez,
                    },
                    normal,
                    last_seen: now,
                    ttl_ms: PLANE_CACHE_TTL_MS,
                    is_hit_patch: false,
                });
            }
        }
    }

    pub fn classify_surface_point(&self, pos: Vec3, normal: Vec3, cam_y: f32) -> SurfaceKind {
        let n = normal.normalize_or_zero();
        if !pos.is_finite() || n.length_squared() < 1e-4 {
            return SurfaceKind::Unknown;
        }
        // Check if an existing remembered plane already covers this point
        for existing in &self.planes {
            if existing.normal.dot(n) < 0.78 {
                continue;
            }
            let rel = pos - existing.marker.pos;
            if rel.dot(existing.normal).abs() > 0.14 {
                continue;
            }
            let ax_x = existing.marker.rot * Vec3::X;
            let ax_z = existing.marker.rot * Vec3::Z;
            if rel.dot(ax_x).abs() <= existing.marker.extent_x * 0.55 + 0.12
                && rel.dot(ax_z).abs() <= existing.marker.extent_z * 0.55 + 0.12
            {
                return existing.marker.kind;
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

    pub fn ingest_hit_surface(
        &mut self,
        pos: Vec3,
        normal: Vec3,
        cam_y: f32,
        hit_kind: i32,
    ) -> SurfaceKind {
        self.ingest_hit_surface_at(pos, normal, cam_y, hit_kind, Instant::now())
    }

    pub fn ingest_hit_surface_at(
        &mut self,
        pos: Vec3,
        normal: Vec3,
        cam_y: f32,
        hit_kind: i32,
        now: Instant,
    ) -> SurfaceKind {
        if hit_kind == 3 {
            return SurfaceKind::Floor;
        }
        let n = normal.normalize_or_zero();
        if !pos.is_finite() || n.length_squared() < 1e-4 {
            return SurfaceKind::Unknown;
        }

        let kind = self.classify_surface_point(pos, n, cam_y);
        if hit_kind != 2 || kind == SurfaceKind::Unknown {
            return kind;
        }

        // Only synthesize a 0.36m x 0.36m patch if no tracked plane covers this surface yet
        let covered = self.planes.iter_mut().any(|old| {
            let ndot = old.normal.dot(n);
            if ndot < 0.84 {
                return false;
            }
            let rel = pos - old.marker.pos;
            if rel.dot(old.normal).abs() > 0.12 {
                return false;
            }
            let ax_x = old.marker.rot * Vec3::X;
            let ax_z = old.marker.rot * Vec3::Z;
            let inside = rel.dot(ax_x).abs() <= old.marker.extent_x * 0.52
                && rel.dot(ax_z).abs() <= old.marker.extent_z * 0.52;
            if inside && old.is_hit_patch {
                old.marker.pos = old.marker.pos * 0.6 + pos * 0.4;
                old.last_seen = now;
            }
            inside
        });

        if !covered {
            let ref_axis = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
            let tx = ref_axis.cross(n).normalize_or_zero();
            let tz = tx.cross(n).normalize_or_zero();
            let rot = Quat::from_mat3(&glam::Mat3::from_cols(tx, n, tz)).normalize();
            if self.planes.len() >= MAX_CACHED_PLANES {
                if let Some((oldest_idx, _)) = self
                    .planes
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, e)| e.last_seen)
                {
                    self.planes.swap_remove(oldest_idx);
                }
            }
            self.planes.push(CachedPlaneEntry {
                marker: PlaneMarker {
                    kind,
                    pos,
                    rot,
                    extent_x: 0.36,
                    extent_z: 0.36,
                },
                normal: n,
                last_seen: now,
                ttl_ms: HIT_PATCH_TTL_MS,
                is_hit_patch: true,
            });
        }
        kind
    }

    pub fn ingest_detected_objects(
        &mut self,
        count: usize,
        packed: &[f32],
        cam_pos: Vec3,
        cam_rot: Quat,
        tan_h: f32,
        tan_v: f32,
        scan_mode: i32,
    ) {
        self.ingest_detected_objects_at(
            count,
            packed,
            cam_pos,
            cam_rot,
            tan_h,
            tan_v,
            scan_mode,
            Instant::now(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ingest_detected_objects_at(
        &mut self,
        count: usize,
        packed: &[f32],
        cam_pos: Vec3,
        cam_rot: Quat,
        tan_h: f32,
        tan_v: f32,
        scan_mode: i32,
        now: Instant,
    ) {
        self.objects
            .retain(|o| now.duration_since(o.last_seen).as_millis() <= OBJECT_CACHE_TTL_MS);

        if scan_mode != 2 || count == 0 {
            return;
        }

        let safe_h = tan_h.clamp(0.2, 2.5);
        let safe_v = tan_v.clamp(0.2, 2.5);
        let markers = parse_detected_objects(count, packed, scan_mode);

        for obj in markers {
            let ray_local = Vec3::new(obj.ndc_x * safe_h, obj.ndc_y * safe_v, -1.0);
            let ray_dir = (cam_rot * ray_local).normalize_or_zero();
            if ray_dir.length_squared() < 1e-4 {
                continue;
            }

            let mut best_t: Option<f32> = None;
            for p in &self.planes {
                let denom = ray_dir.dot(p.normal);
                if denom.abs() <= 0.15 {
                    continue;
                }
                let t = (p.marker.pos - cam_pos).dot(p.normal) / denom;
                if !(0.35..=3.8).contains(&t) {
                    continue;
                }
                let hit = cam_pos + ray_dir * t;
                let rel = hit - p.marker.pos;
                let ax_x = p.marker.rot * Vec3::X;
                let ax_z = p.marker.rot * Vec3::Z;
                let dx = rel.dot(ax_x).abs();
                let dz = rel.dot(ax_z).abs();
                if dx <= p.marker.extent_x * 0.75 + 0.15 && dz <= p.marker.extent_z * 0.75 + 0.15 {
                    if best_t.map(|cur| t < cur).unwrap_or(true) {
                        best_t = Some(t);
                    }
                }
            }

            let depth = best_t.unwrap_or(match obj.kind_id {
                15 | 17 | 19 => 0.85,
                _ => 1.55,
            });
            let world_pos = cam_pos + ray_dir * depth;
            if !world_pos.is_finite() {
                continue;
            }

            let hint = obj.ndc_w.max(obj.ndc_h);
            let existing_idx = self.objects.iter().position(|old| {
                old.marker.kind_id == obj.kind_id
                    && (old.marker.world_pos - world_pos).length() < 0.45
            });

            if let Some(idx) = existing_idx {
                let old = &mut self.objects[idx];
                old.marker.world_pos = old.marker.world_pos * 0.65 + world_pos * 0.35;
                old.marker.score = obj.score.max(old.marker.score * 0.92);
                old.marker.radius_hint_ndc = old.marker.radius_hint_ndc * 0.5 + hint * 0.5;
                old.last_seen = now;
            } else {
                if self.objects.len() >= MAX_CACHED_OBJECTS {
                    if let Some((oldest_idx, _)) = self
                        .objects
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.last_seen)
                    {
                        self.objects.swap_remove(oldest_idx);
                    }
                }
                self.objects.push(CachedObjectEntry {
                    marker: WorldObjectMarker {
                        kind_id: obj.kind_id,
                        world_pos,
                        radius_hint_ndc: hint,
                        score: obj.score,
                    },
                    last_seen: now,
                });
            }
        }
    }

    pub fn snapshot_visible(&mut self, scan_mode: i32, now: Instant) -> (Vec<PlaneMarker>, Vec<WorldObjectMarker>) {
        self.planes
            .retain(|p| now.duration_since(p.last_seen).as_millis() <= p.ttl_ms);
        self.objects
            .retain(|o| now.duration_since(o.last_seen).as_millis() <= OBJECT_CACHE_TTL_MS);

        if scan_mode <= 0 {
            return (Vec::new(), Vec::new());
        }

        let planes = self
            .planes
            .iter()
            .filter_map(|p| {
                let include = if scan_mode == 1 || scan_mode == 3 {
                    matches!(
                        p.marker.kind,
                        SurfaceKind::Floor | SurfaceKind::Wall | SurfaceKind::Table | SurfaceKind::Seat
                    ) || p.is_hit_patch
                } else {
                    p.marker.kind != SurfaceKind::Unknown
                };
                if include {
                    Some(p.marker)
                } else {
                    None
                }
            })
            .collect();

        let objects = if scan_mode == 2 {
            self.objects.iter().map(|o| o.marker).collect()
        } else {
            Vec::new()
        };

        (planes, objects)
    }

    pub fn ingest_point_cloud(&mut self, point_count: usize, packed_xyzc: &[f32], cam_pos: Vec3) {
        self.ingest_point_cloud_at(point_count, packed_xyzc, cam_pos, Instant::now());
    }

    pub fn ingest_point_cloud_at(
        &mut self,
        point_count: usize,
        packed_xyzc: &[f32],
        cam_pos: Vec3,
        now: Instant,
    ) {
        self.voxels.retain(|_, cell| {
            let age = now.duration_since(cell.last_seen).as_millis();
            if cell.hits < 3 {
                age <= VOXEL_NOISE_TTL_MS
            } else {
                age <= VOXEL_TTL_MS
            }
        });

        let n = point_count.min(MAX_POINT_CLOUD_PTS);
        for i in 0..n {
            let base = i * 4;
            if base + 4 > packed_xyzc.len() {
                break;
            }
            let conf = packed_xyzc[base + 3];
            if !conf.is_finite() || conf < 0.35 {
                continue;
            }
            let pt = Vec3::new(packed_xyzc[base], packed_xyzc[base + 1], packed_xyzc[base + 2]);
            if !pt.is_finite() {
                continue;
            }
            let dist_sq = (pt - cam_pos).length_squared();
            if !(0.25 * 0.25..=3.5 * 3.5).contains(&dist_sq) {
                continue;
            }

            let key = (
                (pt.x / VOXEL_SIZE).floor() as i32,
                (pt.y / VOXEL_SIZE).floor() as i32,
                (pt.z / VOXEL_SIZE).floor() as i32,
            );
            if let Some(cell) = self.voxels.get_mut(&key) {
                if cell.hits < 64 {
                    cell.sum_pos += pt;
                    cell.hits += 1;
                } else {
                    let avg = cell.sum_pos / (cell.hits as f32);
                    cell.sum_pos = (avg * 0.85 + pt * 0.15) * (cell.hits as f32);
                }
                cell.last_seen = now;
            } else {
                if self.voxels.len() >= MAX_VOXELS {
                    if let Some(&oldest_key) = self
                        .voxels
                        .iter()
                        .min_by_key(|(_, c)| (c.hits >= 3, c.last_seen))
                        .map(|(k, _)| k)
                    {
                        self.voxels.remove(&oldest_key);
                    }
                }
                self.voxels.insert(
                    key,
                    VoxelCell {
                        sum_pos: pt,
                        hits: 1,
                        last_seen: now,
                    },
                );
            }
        }
    }

    pub fn build_25d_colliders(&self, cam_pos: Vec3, scan_mode: i32) -> Vec<AabbCollider> {
        if scan_mode <= 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(80);

        // Extruded 2.5D object obstacle boxes from ML detections (active in +AI Vật mode 2)
        if scan_mode == 2 {
            for entry in &self.objects {
                let obj = entry.marker;
                let dist = (obj.world_pos - cam_pos).length().clamp(0.4, 4.0);
                let hw = (obj.radius_hint_ndc * 0.28 * dist).clamp(0.06, 0.30);

                // Find highest horizontal supporting surface under object to extrude downward
                let mut support_y: Option<f32> = None;
                for p in &self.planes {
                    if matches!(
                        p.marker.kind,
                        SurfaceKind::Floor | SurfaceKind::Table | SurfaceKind::Seat
                    ) && p.marker.pos.y <= obj.world_pos.y
                        && (obj.world_pos.y - p.marker.pos.y) <= 0.85
                    {
                        let rel = obj.world_pos - p.marker.pos;
                        let dx = rel.dot(p.marker.rot * Vec3::X).abs();
                        let dz = rel.dot(p.marker.rot * Vec3::Z).abs();
                        if dx <= p.marker.extent_x * 0.65 + 0.15
                            && dz <= p.marker.extent_z * 0.65 + 0.15
                            && support_y.map(|cy| p.marker.pos.y > cy).unwrap_or(true)
                        {
                            support_y = Some(p.marker.pos.y);
                        }
                    }
                }

                let min_y = support_y.unwrap_or(obj.world_pos.y - hw);
                let max_y = (obj.world_pos.y + hw).max(min_y + 0.08);
                out.push(AabbCollider {
                    min: Vec3::new(obj.world_pos.x - hw, min_y, obj.world_pos.z - hw),
                    max: Vec3::new(obj.world_pos.x + hw, max_y, obj.world_pos.z + hw),
                    is_object_box: true,
                });
            }
        }

        // Confirmed 2.5D SLAM voxels (excluding flat points flush on continuous horizontal planes)
        let mut candidates: Vec<(f32, Vec3)> = self
            .voxels
            .values()
            .filter_map(|cell| {
                if cell.hits < 3 {
                    return None;
                }
                let center = cell.sum_pos / (cell.hits as f32);
                let d2 = (center - cam_pos).length_squared();
                if d2 > 2.8 * 2.8 {
                    return None;
                }
                let on_flat_plane = self.planes.iter().any(|p| {
                    if !matches!(
                        p.marker.kind,
                        SurfaceKind::Floor | SurfaceKind::Table | SurfaceKind::Seat
                    ) {
                        return false;
                    }
                    let rel = center - p.marker.pos;
                    if rel.y.abs() >= 0.045 {
                        return false;
                    }
                    let dx = rel.dot(p.marker.rot * Vec3::X).abs();
                    let dz = rel.dot(p.marker.rot * Vec3::Z).abs();
                    dx <= p.marker.extent_x * 0.55 && dz <= p.marker.extent_z * 0.55
                });
                if on_flat_plane {
                    None
                } else {
                    Some((d2, center))
                }
            })
            .collect();

        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        let half = Vec3::splat(0.032);
        for (_, center) in candidates.into_iter().take(64) {
            out.push(AabbCollider {
                min: center - half,
                max: center + half,
                is_object_box: false,
            });
        }

        out
    }
}

/// Estimates the room floor Y elevation from tracked upward-facing horizontal ARCore planes.
pub fn infer_floor_y(planes: &[(i32, Vec3, f32, f32)], cam_y: f32) -> Option<f32> {
    let mut min_upward_y: Option<f32> = None;
    let mut max_upward_y: Option<f32> = None;
    let mut min_area = 0.0f32;

    for &(raw_type, pos, ex, ez) in planes {
        if raw_type == 0 {
            let area = (ex * ez).max(0.0);
            match min_upward_y {
                None => {
                    min_upward_y = Some(pos.y);
                    min_area = area;
                }
                Some(prev) if pos.y < prev => {
                    min_upward_y = Some(pos.y);
                    min_area = area;
                }
                _ => {}
            }
            max_upward_y = Some(match max_upward_y {
                None => pos.y,
                Some(prev) => prev.max(pos.y),
            });
        }
    }

    let min_y = min_upward_y?;
    let max_y = max_upward_y.unwrap_or(min_y);
    let below_cam = cam_y - min_y;
    if below_cam >= 0.70 || min_area >= 1.2 || (max_y - min_y) >= 0.30 {
        Some(min_y)
    } else {
        None
    }
}

/// Classifies an ARCore plane into Floor, Wall (size-filtered), Table, Seat, Ceiling, or VerticalObject.
pub fn classify_ar_plane(
    raw_type: i32,
    center_y: f32,
    extent_x: f32,
    extent_z: f32,
    cam_y: f32,
    floor_y: Option<f32>,
) -> SurfaceKind {
    if extent_x.max(extent_z) < 0.22 {
        return SurfaceKind::Unknown;
    }
    match raw_type {
        1 => {
            let max_dim = extent_x.max(extent_z);
            let area = extent_x * extent_z;
            if max_dim >= 0.70 && area >= 0.38 {
                SurfaceKind::Wall
            } else {
                SurfaceKind::VerticalObject
            }
        }
        2 => SurfaceKind::Ceiling,
        0 => {
            if let Some(fy) = floor_y {
                let dy = center_y - fy;
                if dy <= 0.16 {
                    SurfaceKind::Floor
                } else if dy >= 0.28 && dy < 0.58 && extent_x.max(extent_z) <= 1.25 {
                    SurfaceKind::Seat
                } else if dy >= 0.28 && dy <= 1.45 {
                    SurfaceKind::Table
                } else {
                    SurfaceKind::Floor
                }
            } else {
                let below_cam = cam_y - center_y;
                if below_cam >= 0.85 || (extent_x * extent_z) >= 1.6 {
                    SurfaceKind::Floor
                } else {
                    SurfaceKind::Table
                }
            }
        }
        _ => SurfaceKind::Unknown,
    }
}

pub fn parse_detected_objects(count: usize, packed: &[f32], scan_mode: i32) -> Vec<ObjectMarker> {
    if scan_mode != 2 || count == 0 {
        return Vec::new();
    }
    let n = count.min(MAX_OBJECTS);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * FLOATS_PER_OBJECT;
        if base + FLOATS_PER_OBJECT > packed.len() {
            break;
        }
        let kind_id = packed[base].round() as i32;
        let ndc_x = packed[base + 1].clamp(-0.98, 0.98);
        let ndc_y = packed[base + 2].clamp(-0.95, 0.95);
        let ndc_w = packed[base + 3].clamp(0.04, 1.5);
        let ndc_h = packed[base + 4].clamp(0.04, 1.5);
        let score = packed[base + 5].clamp(0.0, 1.0);
        if kind_id >= 11 && ndc_x.is_finite() && ndc_y.is_finite() {
            out.push(ObjectMarker {
                kind_id,
                ndc_x,
                ndc_y,
                ndc_w,
                ndc_h,
                score,
            });
        }
    }
    out
}

fn kind_rgb(kind_id: i32) -> [f32; 3] {
    match kind_id {
        1 => [0.18, 0.86, 0.90], // Floor: cyan-teal
        2 => [0.34, 0.68, 1.00], // Wall: sky blue
        3 => [1.00, 0.72, 0.24], // Table: warm amber
        4 => [0.76, 0.52, 1.00], // Seat: soft violet
        5 => [0.58, 0.65, 0.96], // Ceiling: pale indigo
        6 => [0.60, 0.74, 0.90], // VerticalObject: cool slate blue
        11 | 12 | 13 | 14 => [0.24, 0.94, 0.64], // Furniture: emerald
        15 => [0.24, 0.86, 1.00],                // Screen/TV: cyan
        _ => [1.00, 0.80, 0.30],                 // Other objects: soft gold
    }
}

fn push_quad(
    out: &mut Vec<ScanVertex>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    color: [f32; 4],
) {
    let verts = [p0, p1, p2, p0, p2, p3];
    for p in verts {
        out.push(ScanVertex {
            pos: p.to_array(),
            color,
        });
    }
}

fn push_strip(
    out: &mut Vec<ScanVertex>,
    a: Vec3,
    b: Vec3,
    perp_dir: Vec3,
    half_width: f32,
    color: [f32; 4],
) {
    let off = perp_dir * half_width;
    push_quad(out, a - off, b - off, b + off, a + off, color);
}

fn plane_gravity_aligned_axes(plane: &PlaneMarker, cam_pos: Vec3) -> (Vec3, Vec3, Vec3) {
    let n = (plane.rot * Vec3::Y).normalize_or_zero();
    if n.y.abs() < 0.5 {
        let z = (Vec3::Y - n * n.y).normalize_or_zero();
        let x = z.cross(n).normalize_or_zero();
        if x.length_squared() > 1e-4 && z.length_squared() > 1e-4 {
            return (x, n, z);
        }
    } else {
        let mut to_plane = plane.pos - cam_pos;
        to_plane.y = 0.0;
        let fwd_horiz = to_plane.normalize_or_zero();
        let proj = (fwd_horiz - n * fwd_horiz.dot(n)).normalize_or_zero();
        if proj.length_squared() > 1e-4 {
            let x = n.cross(proj).normalize_or_zero();
            if x.length_squared() > 1e-4 {
                return (x, n, proj);
            }
        }
    }
    (
        (plane.rot * Vec3::X).normalize_or_zero(),
        n,
        (plane.rot * Vec3::Z).normalize_or_zero(),
    )
}

fn push_wire_box(
    out: &mut Vec<ScanVertex>,
    min: Vec3,
    max: Vec3,
    top_col: [f32; 4],
    edge_col: [f32; 4],
    thick: f32,
) {
    // Subtle translucent top cap
    push_quad(
        out,
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
        top_col,
    );

    let c = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    let edges = [
        (0, 1, Vec3::Y),
        (1, 2, Vec3::Y),
        (2, 3, Vec3::Y),
        (3, 0, Vec3::Y),
        (4, 5, Vec3::Z),
        (5, 6, Vec3::X),
        (6, 7, Vec3::Z),
        (7, 4, Vec3::X),
        (0, 4, Vec3::X),
        (1, 5, Vec3::Z),
        (2, 6, Vec3::X),
        (3, 7, Vec3::Z),
    ];
    for (i0, i1, perp) in edges {
        push_strip(out, c[i0], c[i1], perp, thick, edge_col);
    }
}

fn surface_tangent_frame(n: Vec3) -> (Vec3, Vec3) {
    let ref_axis = if n.z.abs() < 0.9 { Vec3::Z } else { Vec3::X };
    let tx = ref_axis.cross(n).normalize_or_zero();
    let tz = n.cross(tx).normalize_or_zero();
    (tx, tz)
}

fn build_plane_grid_vertices(plane: &PlaneMarker, cam_pos: Vec3, out: &mut Vec<ScanVertex>) {
    let ax_y = (plane.rot * Vec3::Y).normalize_or_zero();
    let ax_x = (plane.rot * Vec3::X).normalize_or_zero();
    let ax_z = (plane.rot * Vec3::Z).normalize_or_zero();
    let (ax_x, ax_y, ax_z) = if ax_x.length_squared() > 1e-4
        && ax_y.length_squared() > 1e-4
        && ax_z.length_squared() > 1e-4
    {
        (ax_x, ax_y, ax_z)
    } else {
        plane_gravity_aligned_axes(plane, cam_pos)
    };

    let rgb = kind_rgb(plane.kind as i32);
    let fill_col = [rgb[0], rgb[1], rgb[2], 0.06];
    let grid_col = [rgb[0], rgb[1], rgb[2], 0.22];
    let edge_col = [rgb[0], rgb[1], rgb[2], 0.42];

    let dist = (plane.pos - cam_pos).length().clamp(0.35, 6.0);
    let hx = (plane.extent_x * 0.5).clamp(0.16, 1.30);
    let hz = (plane.extent_z * 0.5).clamp(0.16, 1.30);
    let center = plane.pos + ax_y * 0.003;

    let c0 = center - ax_x * hx - ax_z * hz;
    let c1 = center + ax_x * hx - ax_z * hz;
    let c2 = center + ax_x * hx + ax_z * hz;
    let c3 = center - ax_x * hx + ax_z * hz;
    push_quad(out, c0, c1, c2, c3, fill_col);

    let bw = (dist * 0.0022).clamp(0.0018, 0.0042);
    push_strip(out, c0, c1, ax_z, bw, edge_col);
    push_strip(out, c1, c2, ax_x, bw, edge_col);
    push_strip(out, c2, c3, ax_z, bw, edge_col);
    push_strip(out, c3, c0, ax_x, bw, edge_col);

    // 20cm metric grid lines parallel to Z and X inside the outer boundary
    let gw = (dist * 0.0011).clamp(0.0010, 0.0022);
    let step = 0.20f32;
    let nx = ((hx - 0.04) / step).floor().max(0.0) as i32;
    let nz = ((hz - 0.04) / step).floor().max(0.0) as i32;
    for ix in -nx..=nx {
        if out.len() + 12 > MAX_WORLD_VERTICES {
            return;
        }
        let ox = ax_x * (ix as f32 * step);
        push_strip(out, center + ox - ax_z * hz, center + ox + ax_z * hz, ax_x, gw, grid_col);
    }
    for iz in -nz..=nz {
        if out.len() + 12 > MAX_WORLD_VERTICES {
            return;
        }
        let oz = ax_z * (iz as f32 * step);
        push_strip(out, center - ax_x * hx + oz, center + ax_x * hx + oz, ax_z, gw, grid_col);
    }
}

pub fn build_reticle_vertices(reticle: &ReticleHit, cam_pos: Vec3, out: &mut Vec<ScanVertex>) {
    if !reticle.valid || !reticle.pos.is_finite() || out.len() + 160 > MAX_WORLD_VERTICES {
        return;
    }
    let n = {
        let cand = reticle.normal.normalize_or_zero();
        if cand.length_squared() > 1e-4 {
            cand
        } else {
            Vec3::Y
        }
    };
    let (tx, tz) = surface_tangent_frame(n);
    let center = reticle.pos + n * 0.005;
    let dist = (reticle.pos - cam_pos).length().clamp(0.30, 5.0);
    let r = (dist * 0.044).clamp(0.030, 0.072);
    let hw = (r * 0.10).clamp(0.0026, 0.0055);

    let rgb = if reticle.surface_kind != SurfaceKind::Unknown {
        kind_rgb(reticle.surface_kind as i32)
    } else {
        [0.92, 0.96, 1.00]
    };
    let is_estimate = reticle.hit_kind == 3 || reticle.hit_kind == 4;
    let ring_col = [
        rgb[0],
        rgb[1],
        rgb[2],
        if is_estimate { 0.42 } else { 0.72 },
    ];
    let tick_col = [1.0, 1.0, 1.0, if is_estimate { 0.38 } else { 0.70 }];

    const SEGMENTS: usize = 16;
    for i in 0..SEGMENTS {
        if reticle.hit_kind == 4 && i % 2 == 1 {
            continue; // Dashed preview ring for InstantPlacement estimates (never auto-persisted)
        }
        let a0 = (i as f32) * std::f32::consts::TAU / (SEGMENTS as f32);
        let a1 = ((i + 1) as f32) * std::f32::consts::TAU / (SEGMENTS as f32);
        let d0 = tx * a0.cos() + tz * a0.sin();
        let d1 = tx * a1.cos() + tz * a1.sin();
        push_quad(
            out,
            center + d0 * (r - hw),
            center + d0 * (r + hw),
            center + d1 * (r + hw),
            center + d1 * (r - hw),
            ring_col,
        );
    }

    // Center drop diamond + 4 cardinal ticks
    let cr = (r * 0.17).clamp(0.005, 0.012);
    push_quad(
        out,
        center - tx * cr,
        center - tz * cr,
        center + tx * cr,
        center + tz * cr,
        tick_col,
    );
    let tw = (hw * 0.65).clamp(0.0018, 0.0036);
    push_strip(out, center + tx * (r * 0.48), center + tx * (r * 0.88), tz, tw, tick_col);
    push_strip(out, center - tx * (r * 0.48), center - tx * (r * 0.88), tz, tw, tick_col);
    push_strip(out, center + tz * (r * 0.48), center + tz * (r * 0.88), tx, tw, tick_col);
    push_strip(out, center - tz * (r * 0.48), center - tz * (r * 0.88), tx, tw, tick_col);
}

/// Builds subtle 20cm world-locked metric plane grids, surface-hugging 3D reticle, object badges & 2.5D boxes.
#[allow(clippy::too_many_arguments)]
pub fn build_room_marker_vertices(
    planes: &[PlaneMarker],
    objects: &[WorldObjectMarker],
    blocks_25d: &[AabbCollider],
    reticle: &ReticleHit,
    show_grid: bool,
    cam_pos: Vec3,
    cam_rot: Quat,
    out: &mut Vec<ScanVertex>,
) {
    out.clear();

    if show_grid {
        for plane in planes.iter().take(24) {
            if out.len() + 260 > MAX_WORLD_VERTICES {
                break;
            }
            build_plane_grid_vertices(plane, cam_pos, out);
        }

        let cam_right = (cam_rot * Vec3::X).normalize_or_zero();
        let cam_up = (cam_rot * Vec3::Y).normalize_or_zero();
        let cam_fwd = (cam_rot * -Vec3::Z).normalize_or_zero();

        for obj in objects.iter().take(16) {
            if out.len() + 66 > MAX_WORLD_VERTICES {
                break;
            }
            let rgb = kind_rgb(obj.kind_id);
            let fill_col = [rgb[0], rgb[1], rgb[2], 0.15];
            let tick_col = [rgb[0], rgb[1], rgb[2], 0.56];
            let dot_col = [1.0, 1.0, 1.0, 0.70];

            let dist = (obj.world_pos - cam_pos).length().clamp(0.35, 6.0);
            let b = (dist * 0.055).clamp(0.030, 0.095);
            let dr = (b * 0.40).clamp(0.014, 0.030);
            let center = obj.world_pos;

            push_quad(
                out,
                center + cam_up * dr,
                center + cam_right * dr,
                center - cam_up * dr,
                center - cam_right * dr,
                fill_col,
            );

            let ds = (dist * 0.0040).clamp(0.0035, 0.0075);
            let dot_c = center - cam_fwd * 0.001;
            push_quad(
                out,
                dot_c - cam_right * ds - cam_up * ds,
                dot_c + cam_right * ds - cam_up * ds,
                dot_c + cam_right * ds + cam_up * ds,
                dot_c - cam_right * ds + cam_up * ds,
                dot_col,
            );

            let leg = (b * 0.36).clamp(0.012, 0.030);
            let thick = (dist * 0.0022).clamp(0.0020, 0.0042);
            for (sx, sy) in [(-1.0f32, 1.0f32), (1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
                let corner = center + cam_right * (sx * b) + cam_up * (sy * b);
                push_strip(
                    out,
                    corner,
                    corner - cam_right * (sx * leg),
                    cam_up,
                    thick,
                    tick_col,
                );
                push_strip(
                    out,
                    corner,
                    corner - cam_up * (sy * leg),
                    cam_right,
                    thick,
                    tick_col,
                );
            }
        }

        for block in blocks_25d.iter().filter(|b| b.is_object_box).take(16) {
            if out.len() + 80 > MAX_WORLD_VERTICES {
                break;
            }
            let top_col = [0.24, 0.94, 0.64, 0.08];
            let edge_col = [0.24, 0.94, 0.64, 0.34];
            push_wire_box(out, block.min, block.max, top_col, edge_col, 0.0022);
        }
    }

    build_reticle_vertices(reticle, cam_pos, out);
}

fn project_point_to_uv(
    vp: &Mat4,
    world_pos: Vec3,
    uv_bounds: (f32, f32, f32, f32),
) -> Option<(f32, f32)> {
    let clip = *vp * world_pos.extend(1.0);
    if clip.w <= 0.08 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let u = ndc_x * 0.5 + 0.5;
    let v = 0.5 - ndc_y * 0.5;

    let u_lo = (uv_bounds.0 + 0.025).max(0.04);
    let u_hi = (uv_bounds.1 - 0.025).min(0.96);
    let v_lo = (uv_bounds.2 + 0.035).max(0.14);
    let v_hi = (uv_bounds.3 - 0.035).min(0.90);

    if u >= u_lo && u <= u_hi && v >= v_lo && v <= v_hi {
        Some((u, v))
    } else {
        None
    }
}

/// Projects up to `MAX_LABELS` non-overlapping stereo labels constrained inside the camera viewport.
pub fn project_room_labels(
    planes: &[PlaneMarker],
    objects: &[WorldObjectMarker],
    left_vp: &Mat4,
    right_vp: &Mat4,
    left_uv_bounds: (f32, f32, f32, f32),
    right_uv_bounds: (f32, f32, f32, f32),
) -> Vec<ProjectedLabel> {
    let mut out: Vec<ProjectedLabel> = Vec::with_capacity(MAX_LABELS);

    let try_push = |out: &mut Vec<ProjectedLabel>, kind_id: i32, world_pos: Vec3, score: f32| {
        if out.len() >= MAX_LABELS {
            return;
        }
        let Some((lu, lv)) = project_point_to_uv(left_vp, world_pos, left_uv_bounds) else {
            return;
        };
        let Some((ru, rv)) = project_point_to_uv(right_vp, world_pos, right_uv_bounds) else {
            return;
        };
        let mu = (lu + ru) * 0.5;
        let mv = (lv + rv) * 0.5;
        let overlaps = out.iter().any(|existing| {
            let eu = (existing.left_u + existing.right_u) * 0.5;
            let ev = (existing.left_v + existing.right_v) * 0.5;
            let du = mu - eu;
            let dv = mv - ev;
            (du * du + dv * dv).sqrt() < 0.14
        });
        if !overlaps {
            out.push(ProjectedLabel {
                kind_id,
                left_u: lu,
                left_v: lv,
                right_u: ru,
                right_v: rv,
                score,
            });
        }
    };

    let mut sorted_objs: Vec<&WorldObjectMarker> = objects.iter().collect();
    sorted_objs.sort_by(|a, b| b.score.total_cmp(&a.score));
    for obj in sorted_objs {
        try_push(&mut out, obj.kind_id, obj.world_pos, obj.score);
    }

    let mut sorted_planes: Vec<&PlaneMarker> = planes.iter().collect();
    sorted_planes.sort_by_key(|p| match p.kind {
        SurfaceKind::Table => 0,
        SurfaceKind::Seat => 1,
        SurfaceKind::Floor => 2,
        SurfaceKind::Wall => 3,
        SurfaceKind::VerticalObject => 4,
        SurfaceKind::Ceiling => 5,
        SurfaceKind::Unknown => 6,
    });
    for plane in sorted_planes {
        let score = match plane.kind {
            SurfaceKind::Floor => 0.92,
            SurfaceKind::Wall => 0.88,
            SurfaceKind::Table => 0.85,
            SurfaceKind::Seat => 0.80,
            SurfaceKind::Ceiling => 0.75,
            SurfaceKind::VerticalObject => 0.68,
            SurfaceKind::Unknown => 0.0,
        };
        try_push(&mut out, plane.kind as i32, plane.pos, score);
    }

    out
}

pub struct RoomScannerRenderer {
    program: glow::Program,
    world_vao: glow::VertexArray,
    world_vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    world_cpu: Vec<ScanVertex>,
    world_vert_count: i32,
}

impl RoomScannerRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, SCAN_VERT, SCAN_FRAG)?;
            let world_vao = gl.create_vertex_array()?;
            let world_vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(world_vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(world_vbo));
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                (MAX_WORLD_VERTICES * size_of::<ScanVertex>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<ScanVertex>() as i32;
            if let Some(a_pos) = gl.get_attrib_location(program, "a_pos") {
                gl.enable_vertex_attrib_array(a_pos);
                gl.vertex_attrib_pointer_f32(a_pos, 3, glow::FLOAT, false, stride, 0);
            }
            if let Some(a_color) = gl.get_attrib_location(program, "a_color") {
                gl.enable_vertex_attrib_array(a_color);
                gl.vertex_attrib_pointer_f32(
                    a_color,
                    4,
                    glow::FLOAT,
                    false,
                    stride,
                    (3 * size_of::<f32>()) as i32,
                );
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            Ok(Self {
                program,
                world_vao,
                world_vbo,
                u_mvp,
                world_cpu: Vec::with_capacity(MAX_WORLD_VERTICES),
                world_vert_count: 0,
            })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        gl: &glow::Context,
        planes: &[PlaneMarker],
        objects: &[WorldObjectMarker],
        blocks_25d: &[AabbCollider],
        reticle: &ReticleHit,
        show_grid: bool,
        cam_pos: Vec3,
        cam_rot: Quat,
    ) {
        if (!show_grid || (planes.is_empty() && objects.is_empty() && blocks_25d.is_empty()))
            && !reticle.valid
        {
            self.world_vert_count = 0;
            return;
        }
        build_room_marker_vertices(
            planes,
            objects,
            blocks_25d,
            reticle,
            show_grid,
            cam_pos,
            cam_rot,
            &mut self.world_cpu,
        );
        self.world_vert_count = self.world_cpu.len() as i32;
        if self.world_vert_count > 0 {
            unsafe {
                let bytes = std::slice::from_raw_parts(
                    self.world_cpu.as_ptr() as *const u8,
                    self.world_cpu.len() * size_of::<ScanVertex>(),
                );
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.world_vbo));
                gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                gl.bind_buffer(glow::ARRAY_BUFFER, None);
            }
        }
    }

    pub fn draw_world_markers(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.world_vert_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.world_vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.world_vert_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.world_vbo);
            gl.delete_vertex_array(self.world_vao);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn filters_small_vertical_pc_cases_vs_true_walls() {
        let cam_y = 1.45;
        let summary = vec![
            (0, Vec3::new(0.0, 0.0, -2.0), 2.5, 2.5),
            (0, Vec3::new(0.4, 0.44, -1.6), 0.55, 0.55),
            (0, Vec3::new(-0.3, 0.76, -1.5), 1.4, 0.8),
        ];
        let floor_y = infer_floor_y(&summary, cam_y);
        assert_eq!(floor_y, Some(0.0));

        assert_eq!(classify_ar_plane(0, 0.02, 2.5, 2.5, cam_y, floor_y), SurfaceKind::Floor);
        assert_eq!(classify_ar_plane(0, 0.44, 0.55, 0.55, cam_y, floor_y), SurfaceKind::Seat);
        assert_eq!(classify_ar_plane(0, 0.76, 1.4, 0.8, cam_y, floor_y), SurfaceKind::Table);
        // Large vertical plane is a Wall
        assert_eq!(classify_ar_plane(1, 1.2, 1.8, 1.5, cam_y, floor_y), SurfaceKind::Wall);
        // Small vertical plane (PC case / monitor back) is VerticalObject, NOT Wall
        assert_eq!(classify_ar_plane(1, 0.6, 0.45, 0.40, cam_y, floor_y), SurfaceKind::VerticalObject);
        // Tiny artifact < 22cm is ignored
        assert_eq!(classify_ar_plane(1, 0.6, 0.15, 0.18, cam_y, floor_y), SurfaceKind::Unknown);
        assert_eq!(classify_ar_plane(2, 2.7, 2.5, 2.5, cam_y, floor_y), SurfaceKind::Ceiling);
    }

    #[test]
    fn spatial_cache_merges_persists_planes_and_pins_3d_objects() {
        let mut cache = SpatialMemoryCache::default();
        let t0 = Instant::now();
        let packed_plane = [
            0.0, 0.0, -1.2, -1.8, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, // Floor at y=-1.2
            1.0, 0.0, 0.2, -2.5, 0.7071, 0.0, 0.0, 0.7071, 2.0, 1.6, // Large Wall at z=-2.5
            1.0, -0.5, -0.4, -0.9, 0.7071, 0.0, 0.0, 0.7071, 0.4, 0.35, // Small PC case
        ];
        cache.ingest_ar_planes_at(3, &packed_plane, 0.0, 2, t0);
        assert_eq!(cache.cached_plane_count(), 3);

        // Mode 1 (Sàn/Tường) filters out small VerticalObject
        let (vis_m1, _) = cache.snapshot_visible(1, t0 + Duration::from_secs(5));
        assert_eq!(vis_m1.len(), 2);

        // Pin detected Monitor object via center ray onto Floor/Wall or fallback depth
        let packed_obj = [15.0, 0.0, 0.0, 0.3, 0.3, 0.85];
        cache.ingest_detected_objects_at(1, &packed_obj, Vec3::ZERO, Quat::IDENTITY, 0.70, 0.55, 2, t0);
        assert_eq!(cache.cached_object_count(), 1);

        // Remembered after 15s for planes, expired after 11s for objects
        let (vis_planes_15s, vis_objs_15s) = cache.snapshot_visible(2, t0 + Duration::from_secs(15));
        assert_eq!(vis_planes_15s.len(), 3);
        assert_eq!(vis_objs_15s.len(), 0);
    }

    #[test]
    fn metric_grid_stays_faint_world_locked_and_reticle_renders_when_grid_hidden() {
        let tilted_rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2) * Quat::from_rotation_y(0.7);
        let planes = vec![PlaneMarker {
            kind: SurfaceKind::Table,
            pos: Vec3::new(0.0, -0.4, -1.0),
            rot: Quat::IDENTITY,
            extent_x: 1.4,
            extent_z: 1.0,
        }, PlaneMarker {
            kind: SurfaceKind::Wall,
            pos: Vec3::new(0.0, 0.2, -1.8),
            rot: tilted_rot,
            extent_x: 4.0,
            extent_z: 4.0,
        }];
        let reticle = ReticleHit {
            valid: true,
            hit_kind: 1,
            surface_kind: SurfaceKind::Table,
            pos: Vec3::new(0.0, -0.4, -0.8),
            normal: Vec3::Y,
            distance: 0.89,
        };
        let mut verts = Vec::new();
        build_room_marker_vertices(
            &planes,
            &[],
            &[],
            &reticle,
            true,
            Vec3::ZERO,
            Quat::IDENTITY,
            &mut verts,
        );
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 3, 0);
        assert!(verts.iter().all(|v| v.color[3] <= 0.72));
        // Clamped half-extents (<= 1.30m) prevent runaway vertices even on 4m x 4m walls
        assert!(verts.len() < MAX_WORLD_VERTICES);

        // Mode 3 (show_grid = false): only surface-hugging reticle vertices are emitted
        let mut reticle_only = Vec::new();
        build_room_marker_vertices(
            &planes,
            &[],
            &[],
            &reticle,
            false,
            Vec3::ZERO,
            Quat::IDENTITY,
            &mut reticle_only,
        );
        assert!(!reticle_only.is_empty());
        assert!(reticle_only.len() < 160);
        assert!(reticle_only.iter().all(|v| (v.pos[1] - (-0.4 + 0.005)).abs() < 1e-4));
    }

    #[test]
    fn ingest_hit_surface_synthesizes_patch_and_expires_after_20s() {
        let mut cache = SpatialMemoryCache::default();
        let t0 = Instant::now();
        let table_pos = Vec3::new(0.1, -0.55, -0.75);
        let kind = cache.ingest_hit_surface_at(table_pos, Vec3::Y, 0.0, 2, t0);
        assert_eq!(kind, SurfaceKind::Table);
        assert_eq!(cache.cached_plane_count(), 1);

        let (planes_now, _) = cache.snapshot_visible(1, t0 + Duration::from_secs(5));
        assert_eq!(planes_now.len(), 1);
        assert!((planes_now[0].extent_x - 0.36).abs() < 1e-4);

        // Hitting nearby refreshes without duplicating
        let kind2 = cache.ingest_hit_surface_at(table_pos + Vec3::new(0.05, 0.0, 0.04), Vec3::Y, 0.0, 2, t0 + Duration::from_secs(6));
        assert_eq!(kind2, SurfaceKind::Table);
        assert_eq!(cache.cached_plane_count(), 1);

        // Expires after 21s without refresh
        let (planes_expired, _) = cache.snapshot_visible(1, t0 + Duration::from_secs(28));
        assert_eq!(planes_expired.len(), 0);
    }

    #[test]
    fn point_cloud_builds_25d_voxels_prunes_noise_and_extrudes_objects() {
        let mut cache = SpatialMemoryCache::default();
        let t0 = Instant::now();
        // Confirmed obstacle cluster (3 high-confidence hits inside cell x:[0.12..0.18], y:[-0.36..-0.30], z:[-1.26..-1.20])
        let pts_cluster = [
            0.14, -0.33, -1.22, 0.85,
            0.15, -0.34, -1.23, 0.80,
            0.16, -0.32, -1.24, 0.90,
            // Single transient noisy hit + low confidence hit
            -0.50, -0.10, -1.40, 0.75,
            0.00, 0.00, -1.00, 0.15,
        ];
        cache.ingest_point_cloud_at(5, &pts_cluster, Vec3::ZERO, t0);
        assert_eq!(cache.confirmed_voxel_count(), 1);

        // Add detected monitor above table -> extrudes to 2.5D collider
        let packed_obj = [15.0, 0.0, 0.0, 0.35, 0.35, 0.85];
        cache.ingest_detected_objects_at(1, &packed_obj, Vec3::ZERO, Quat::IDENTITY, 0.70, 0.55, 2, t0);

        let aabbs = cache.build_25d_colliders(Vec3::ZERO, 2);
        assert!(aabbs.len() >= 2);
        assert!(aabbs.iter().any(|a| a.is_object_box));
        assert!(aabbs.iter().any(|a| !a.is_object_box));

        // After 10s without refresh, transient 1-hit noise prunes while confirmed cluster remains within 35s TTL
        cache.ingest_point_cloud_at(0, &[], Vec3::ZERO, t0 + Duration::from_secs(10));
        assert_eq!(cache.confirmed_voxel_count(), 1);
    }

    #[test]
    fn labels_clip_strictly_inside_camera_quad_bounds() {
        let proj = Mat4::perspective_rh_gl(85.0_f32.to_radians(), 1200.0 / 1080.0, 0.05, 50.0);
        let left_vp = proj * Mat4::from_translation(Vec3::new(0.032, 0.0, 0.0));
        let right_vp = proj * Mat4::from_translation(Vec3::new(-0.032, 0.0, 0.0));

        let planes = vec![PlaneMarker {
            kind: SurfaceKind::Table,
            pos: Vec3::new(0.0, -0.45, -1.6),
            rot: Quat::IDENTITY,
            extent_x: 1.0,
            extent_z: 0.8,
        }];
        let objs = vec![WorldObjectMarker {
            kind_id: 12,
            world_pos: Vec3::new(0.0, 0.0, -1.6),
            radius_hint_ndc: 0.25,
            score: 0.85,
        }];

        let bounds = (0.15, 0.85, 0.20, 0.80);
        let labels = project_room_labels(&planes, &objs, &left_vp, &right_vp, bounds, bounds);
        assert!(!labels.is_empty());
        for lbl in &labels {
            assert!((0.15..=0.85).contains(&lbl.left_u));
            assert!((0.20..=0.80).contains(&lbl.left_v));
            assert!(lbl.right_u < lbl.left_u);
        }
    }
}
