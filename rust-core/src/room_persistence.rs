use crate::room_scanner::SurfaceKind;
use crate::spatial_mesh::{
    CanonicalPlane, SavedTsdfVoxel, SpatialMeshMap, CELL_SIZE, CHUNK_DIM, MAX_POLY_VERTS,
};
use crate::vision_ram::{RELOC_DESC_H, RELOC_DESC_LEN, RELOC_DESC_W};
use glam::{Quat, Vec3};
use std::fmt::Write as _;

pub const ARROOM_MAGIC: [u8; 4] = *b"ARRM";
pub const ARROOM_VERSION: u16 = 2; // v2 stores narrow-band TSDF (idx, tsdf: i8, weight: u8); loader stays v1-compatible
pub const MAX_KEYFRAMES: usize = 16;

#[derive(Clone, Debug)]
pub struct RelocKeyframe {
    pub cam_pos: Vec3,
    pub cam_rot: Quat,
    pub desc: [u8; RELOC_DESC_LEN],
}

#[derive(Default, Clone, Debug)]
pub struct KeyframeBank {
    pub frames: Vec<RelocKeyframe>,
}

impl KeyframeBank {
    pub fn clear(&mut self) {
        self.frames.clear();
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Captures a new visual keyframe if the camera moved >= 0.55m or rotated >= 32°
    /// from existing viewpoints and the frame has sufficient visual contrast.
    pub fn maybe_capture(&mut self, cam_pos: Vec3, cam_rot: Quat, desc: [u8; RELOC_DESC_LEN]) -> bool {
        if !cam_pos.is_finite() || cam_rot.length_squared() < 1e-4 {
            return false;
        }
        if descriptor_std_dev(&desc) < 8.0 {
            return false;
        }
        let rot = cam_rot.normalize();
        let too_close = self.frames.iter().any(|k| {
            let dp = (k.cam_pos - cam_pos).length();
            let da = k.cam_rot.angle_between(rot);
            dp < 0.45 && da < 0.45
        });
        if too_close {
            return false;
        }

        if self.frames.len() >= MAX_KEYFRAMES {
            // Preserve frame #0 as primary room anchor; rotate slots 1..MAX_KEYFRAMES
            self.frames.remove(1);
        }
        self.frames.push(RelocKeyframe {
            cam_pos,
            cam_rot: rot,
            desc,
        });
        true
    }

    pub fn transform_and_merge(&mut self, saved: &[RelocKeyframe], q_align: Quat, t_align: Vec3) {
        for kf in saved {
            if self.frames.len() >= MAX_KEYFRAMES {
                break;
            }
            self.frames.push(RelocKeyframe {
                cam_pos: q_align * kf.cam_pos + t_align,
                cam_rot: (q_align * kf.cam_rot).normalize(),
                desc: kf.desc,
            });
        }
    }
}

#[derive(Clone, Debug)]
pub struct LoadedRoomMap {
    pub floor_y: Option<f32>,
    pub planes: Vec<CanonicalPlane>,
    pub tsdf_voxels_world: Vec<SavedTsdfVoxel>,
    pub keyframes: Vec<RelocKeyframe>,
    pub consecutive_hits: u8,
    pub last_matched_kf: Option<usize>,
    pub last_best_score: f32,
}

fn descriptor_std_dev(desc: &[u8; RELOC_DESC_LEN]) -> f32 {
    let n = RELOC_DESC_LEN as f32;
    let mean = desc.iter().map(|&v| v as f32).sum::<f32>() / n;
    let var = desc
        .iter()
        .map(|&v| {
            let d = v as f32 - mean;
            d * d
        })
        .sum::<f32>()
        / n;
    var.sqrt()
}

/// Computes camera pitch angle (radians, 0 = level horizon) from rotation quaternion.
fn camera_pitch_rad(rot: Quat) -> f32 {
    let fwd = (rot * -Vec3::Z).normalize_or_zero();
    fwd.y.clamp(-1.0, 1.0).asin()
}

/// Computes horizontal yaw angle `atan2(fwd.x, fwd.z)` so `Quat::from_rotation_y(yaw_c - yaw_s) * fwd_s == fwd_c`.
fn camera_horiz_yaw(rot: Quat) -> f32 {
    let raw = rot * -Vec3::Z;
    let horiz = Vec3::new(raw.x, 0.0, raw.z);
    if horiz.length_squared() > 1e-5 {
        horiz.x.atan2(horiz.z)
    } else {
        let up_raw = rot * Vec3::Y;
        up_raw.x.atan2(up_raw.z)
    }
}

/// Rule 6: Computes exact gravity-locked 4-DoF rigid alignment `(q_align, t_align)` mapping saved
/// session coordinates into the live ARCore session (`p_curr = q_align * p_saved + t_align`).
pub fn compute_4dof_alignment(
    saved_pos: Vec3,
    saved_rot: Quat,
    curr_pos: Vec3,
    curr_rot: Quat,
    saved_floor_y: Option<f32>,
    curr_floor_y: Option<f32>,
) -> (Quat, Vec3) {
    let yaw_s = camera_horiz_yaw(saved_rot);
    let yaw_c = camera_horiz_yaw(curr_rot);
    let q_align = Quat::from_rotation_y(yaw_c - yaw_s);
    let mut t_align = curr_pos - q_align * saved_pos;
    if let (Some(fy_s), Some(fy_c)) = (saved_floor_y, curr_floor_y) {
        t_align.y = fy_c - fy_s;
    }
    (q_align, t_align)
}

/// Brightness-invariant Normalized Cross-Correlation (NCC) with ±2 column shift tolerance.
pub fn descriptor_ncc_shifted(a: &[u8; RELOC_DESC_LEN], b: &[u8; RELOC_DESC_LEN]) -> f32 {
    let mut best = -1.0f32;
    for shift in -2i32..=2i32 {
        let x_start = shift.max(0) as usize;
        let x_end = (RELOC_DESC_W as i32 + shift.min(0)) as usize;
        let cols = x_end.saturating_sub(x_start);
        if cols < 8 {
            continue;
        }
        let count = (cols * RELOC_DESC_H) as f32;

        let mut sum_a = 0.0f32;
        let mut sum_b = 0.0f32;
        for y in 0..RELOC_DESC_H {
            let row = y * RELOC_DESC_W;
            for x in x_start..x_end {
                let bx = (x as i32 - shift) as usize;
                sum_a += a[row + x] as f32;
                sum_b += b[row + bx] as f32;
            }
        }
        let mean_a = sum_a / count;
        let mean_b = sum_b / count;

        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for y in 0..RELOC_DESC_H {
            let row = y * RELOC_DESC_W;
            for x in x_start..x_end {
                let bx = (x as i32 - shift) as usize;
                let da = a[row + x] as f32 - mean_a;
                let db = b[row + bx] as f32 - mean_b;
                dot += da * db;
                norm_a += da * da;
                norm_b += db * db;
            }
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom > 1e-3 {
            best = best.max(dot / denom);
        }
    }
    best
}

/// Evaluates live grayscale frame against saved keyframes; returns `Some((kf_idx, score))`
/// once the same keyframe exceeds NCC >= 0.72 for 2 consecutive checks.
pub fn try_auto_relocalize(
    pending: &mut LoadedRoomMap,
    live_rot: Quat,
    live_desc: &[u8; RELOC_DESC_LEN],
) -> Option<(usize, f32)> {
    if pending.keyframes.is_empty() || descriptor_std_dev(live_desc) < 8.0 {
        return None;
    }
    let live_pitch = camera_pitch_rad(live_rot);
    let mut best_idx: Option<usize> = None;
    let mut best_score = -1.0f32;

    for (idx, kf) in pending.keyframes.iter().enumerate() {
        let pitch_diff = (camera_pitch_rad(kf.cam_rot) - live_pitch).abs();
        if pitch_diff > 0.50 {
            continue;
        }
        let score = descriptor_ncc_shifted(live_desc, &kf.desc);
        if score > best_score {
            best_score = score;
            best_idx = Some(idx);
        }
    }

    pending.last_best_score = best_score.max(0.0);
    if let Some(idx) = best_idx {
        if best_score >= 0.72 {
            if pending.last_matched_kf == Some(idx) {
                pending.consecutive_hits = pending.consecutive_hits.saturating_add(1);
            } else {
                pending.last_matched_kf = Some(idx);
                pending.consecutive_hits = 1;
            }
            if pending.consecutive_hits >= 2 {
                return Some((idx, best_score));
            }
            return None;
        }
    }
    pending.consecutive_hits = 0;
    pending.last_matched_kf = None;
    None
}

/// Picks the best saved keyframe for 1-Tap Manual Relocalize (closest pitch to current view, or anchor #0).
pub fn pick_manual_anchor_keyframe(pending: &LoadedRoomMap, live_rot: Quat) -> Option<usize> {
    if pending.keyframes.is_empty() {
        return None;
    }
    let live_pitch = camera_pitch_rad(live_rot);
    pending
        .keyframes
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (camera_pitch_rad(a.cam_rot) - live_pitch).abs();
            let db = (camera_pitch_rad(b.cam_rot) - live_pitch).abs();
            da.total_cmp(&db)
        })
        .map(|(i, _)| i)
}

fn kind_to_u8(kind: SurfaceKind) -> u8 {
    match kind {
        SurfaceKind::Unknown => 0,
        SurfaceKind::Floor => 1,
        SurfaceKind::Wall => 2,
        SurfaceKind::Table => 3,
        SurfaceKind::Seat => 4,
        SurfaceKind::Ceiling => 5,
        SurfaceKind::VerticalObject => 6,
    }
}

fn u8_to_kind(v: u8) -> SurfaceKind {
    match v {
        1 => SurfaceKind::Floor,
        2 => SurfaceKind::Wall,
        3 => SurfaceKind::Table,
        4 => SurfaceKind::Seat,
        5 => SurfaceKind::Ceiling,
        6 => SurfaceKind::VerticalObject,
        _ => SurfaceKind::Unknown,
    }
}

/// Rule 5: Serializes canonical merged planes, sparse TSDF surface chunks, and visual relocalization
/// keyframes into a compact `.arroom` v2 binary byte buffer.
pub fn serialize_room_arroom(map: &SpatialMeshMap, kf_bank: &KeyframeBank) -> Vec<u8> {
    let tsdf_chunks = map.export_tsdf_chunks();
    let mut buf = Vec::with_capacity(24 * 1024);

    // Header (28 bytes)
    buf.extend_from_slice(&ARROOM_MAGIC);
    buf.extend_from_slice(&ARROOM_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // flags
    buf.extend_from_slice(&CELL_SIZE.to_le_bytes());
    let fy = map.inferred_floor_y().unwrap_or(f32::NAN);
    buf.extend_from_slice(&fy.to_le_bytes());
    buf.extend_from_slice(&(map.planes.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(tsdf_chunks.len() as u32).to_le_bytes());
    buf.extend_from_slice(&(kf_bank.frames.len() as u32).to_le_bytes());

    // Canonical merged planes
    for p in &map.planes {
        let poly_len = p.polygon_xz.len().min(MAX_POLY_VERTS) as u16;
        buf.push(kind_to_u8(p.kind));
        buf.push(p.raw_type as i8 as u8);
        buf.extend_from_slice(&poly_len.to_le_bytes());
        for f in [
            p.pos.x, p.pos.y, p.pos.z, p.rot.x, p.rot.y, p.rot.z, p.rot.w, p.area, p.weight,
        ] {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        for &[x, z] in p.polygon_xz.iter().take(poly_len as usize) {
            buf.extend_from_slice(&x.to_le_bytes());
            buf.extend_from_slice(&z.to_le_bytes());
        }
    }

    // Sparse narrow-band TSDF chunks (4 bytes per surface-band voxel: idx:u16 + tsdf:i8 + weight:u8)
    for (ckey, entries) in &tsdf_chunks {
        buf.extend_from_slice(&ckey.0.to_le_bytes());
        buf.extend_from_slice(&ckey.1.to_le_bytes());
        buf.extend_from_slice(&ckey.2.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for &(idx, tsdf, weight) in entries {
            buf.extend_from_slice(&idx.to_le_bytes());
            buf.push(tsdf as u8);
            buf.push(weight);
        }
    }

    // Visual relocalization keyframes
    for kf in &kf_bank.frames {
        for f in [
            kf.cam_pos.x,
            kf.cam_pos.y,
            kf.cam_pos.z,
            kf.cam_rot.x,
            kf.cam_rot.y,
            kf.cam_rot.z,
            kf.cam_rot.w,
        ] {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf.extend_from_slice(&kf.desc);
    }

    buf
}

struct BinReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], String> {
        if self.pos + n > self.data.len() {
            return Err("Unexpected EOF in .arroom file".to_string());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        let b = self.read_bytes(4)?;
        Ok(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let b = self.read_bytes(4)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Deserializes a `.arroom` v1 or v2 binary buffer with strict bounds validation.
pub fn deserialize_room_arroom(data: &[u8]) -> Result<LoadedRoomMap, String> {
    let mut r = BinReader::new(data);
    let magic = r.read_bytes(4)?;
    if magic != ARROOM_MAGIC {
        return Err("Invalid .arroom magic header".to_string());
    }
    let version = r.read_u16()?;
    if version != 1 && version != ARROOM_VERSION {
        return Err(format!("Unsupported .arroom version: {}", version));
    }
    let _flags = r.read_u16()?;
    let cell_size = r.read_f32()?;
    if !(0.02..=0.25).contains(&cell_size) {
        return Err("Invalid cell_size in .arroom header".to_string());
    }
    let fy_raw = r.read_f32()?;
    let floor_y = if fy_raw.is_finite() { Some(fy_raw) } else { None };
    let plane_count = r.read_u32()? as usize;
    let chunk_count = r.read_u32()? as usize;
    let kf_count = r.read_u32()? as usize;

    if plane_count > 64 || chunk_count > 2048 || kf_count > 32 {
        return Err("Exceeded safety limits in .arroom file".to_string());
    }

    let mut planes = Vec::with_capacity(plane_count);
    for _ in 0..plane_count {
        let kind = u8_to_kind(r.read_u8()?);
        let raw_type = r.read_u8()? as i8 as i32;
        let poly_len = (r.read_u16()? as usize).min(MAX_POLY_VERTS);
        let pos = Vec3::new(r.read_f32()?, r.read_f32()?, r.read_f32()?);
        let rot_raw = Quat::from_xyzw(r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?);
        let rot = if rot_raw.length_squared() > 1e-6 {
            rot_raw.normalize()
        } else {
            Quat::IDENTITY
        };
        let area = r.read_f32()?;
        let weight = r.read_f32()?;
        let mut poly = Vec::with_capacity(poly_len);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_z = f32::INFINITY;
        let mut max_z = f32::NEG_INFINITY;
        for _ in 0..poly_len {
            let x = r.read_f32()?;
            let z = r.read_f32()?;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_z = min_z.min(z);
            max_z = max_z.max(z);
            poly.push([x, z]);
        }
        let ex = if poly_len >= 3 { (max_x - min_x).max(0.1) } else { 0.36 };
        let ez = if poly_len >= 3 { (max_z - min_z).max(0.1) } else { 0.36 };
        let normal = (rot * Vec3::Y).normalize_or_zero();
        if pos.is_finite() && normal.length_squared() > 1e-4 {
            planes.push(CanonicalPlane {
                kind,
                raw_type,
                pos,
                rot,
                normal,
                polygon_xz: poly,
                extent_x: ex,
                extent_z: ez,
                area,
                weight,
                last_seen_gen: 0,
                is_explicit_patch: false,
                carve_hits: 0,
                detail_hits: 0,
                demoted_to_object: false,
            });
        }
    }

    let mut tsdf_voxels_world = Vec::new();
    let d_i = CHUNK_DIM as i32;
    for _ in 0..chunk_count {
        let cx = r.read_i32()?;
        let cy = r.read_i32()?;
        let cz = r.read_i32()?;
        let entries = r.read_u16()? as usize;
        if entries > CHUNK_DIM * CHUNK_DIM * CHUNK_DIM {
            return Err("Corrupt chunk voxel entry count".to_string());
        }
        let base_x = cx * d_i;
        let base_y = cy * d_i;
        let base_z = cz * d_i;
        for _ in 0..entries {
            let idx = r.read_u16()? as usize;
            let (tsdf, weight) = if version >= 2 {
                let t = r.read_u8()? as i8;
                let w = r.read_u8()?;
                (t, w)
            } else {
                let w = r.read_u8()?;
                (-36i8, w.max(36))
            };
            let lx = idx % CHUNK_DIM;
            let lz = (idx / CHUNK_DIM) % CHUNK_DIM;
            let ly = idx / (CHUNK_DIM * CHUNK_DIM);
            if ly < CHUNK_DIM {
                let wx = (base_x + lx as i32) as f32 * cell_size + cell_size * 0.5;
                let wy = (base_y + ly as i32) as f32 * cell_size + cell_size * 0.5;
                let wz = (base_z + lz as i32) as f32 * cell_size + cell_size * 0.5;
                tsdf_voxels_world.push(SavedTsdfVoxel {
                    world_pos: Vec3::new(wx, wy, wz),
                    tsdf,
                    weight,
                });
            }
        }
    }

    let mut keyframes = Vec::with_capacity(kf_count);
    for _ in 0..kf_count {
        let cam_pos = Vec3::new(r.read_f32()?, r.read_f32()?, r.read_f32()?);
        let q = Quat::from_xyzw(r.read_f32()?, r.read_f32()?, r.read_f32()?, r.read_f32()?);
        let cam_rot = if q.length_squared() > 1e-6 {
            q.normalize()
        } else {
            Quat::IDENTITY
        };
        let bytes = r.read_bytes(RELOC_DESC_LEN)?;
        let mut desc = [0u8; RELOC_DESC_LEN];
        desc.copy_from_slice(bytes);
        keyframes.push(RelocKeyframe {
            cam_pos,
            cam_rot,
            desc,
        });
    }

    Ok(LoadedRoomMap {
        floor_y,
        planes,
        tsdf_voxels_world,
        keyframes,
        consecutive_hits: 0,
        last_matched_kf: None,
        last_best_score: 0.0,
    })
}

/// Exports the entire reconstructed 3D room (planes + greedy-meshed obstacles) to standard Wavefront `.obj`.
pub fn export_room_to_obj(map: &mut SpatialMeshMap) -> String {
    let groups = map.build_obj_mesh_groups();
    let mut out = String::with_capacity(32 * 1024);
    let _ = writeln!(out, "# Redmi Note 14 AR/VR — Reconstructed 3D Room (.obj)");
    let mut v_idx: usize = 1;

    for (name, verts) in groups {
        let _ = writeln!(out, "\ng {}", name);
        for tri in verts.chunks_exact(3) {
            for v in tri {
                let _ = writeln!(out, "v {:.4} {:.4} {:.4}", v.pos[0], v.pos[1], v.pos[2]);
            }
            let n = tri[0].normal;
            let _ = writeln!(out, "vn {:.4} {:.4} {:.4}", n[0], n[1], n[2]);
            let vn_i = (v_idx - 1) / 3 + 1;
            let _ = writeln!(
                out,
                "f {}//{} {}//{} {}//{}",
                v_idx,
                vn_i,
                v_idx + 1,
                vn_i,
                v_idx + 2,
                vn_i
            );
            v_idx += 3;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_mesh::RawArPlane;

    fn make_synthetic_desc(seed: f32, brightness_shift: f32) -> [u8; RELOC_DESC_LEN] {
        let mut d = [0u8; RELOC_DESC_LEN];
        for y in 0..RELOC_DESC_H {
            for x in 0..RELOC_DESC_W {
                let val = 110.0
                    + 65.0 * ((x as f32 * 0.35 + seed).sin() * (y as f32 * 0.25).cos())
                    + brightness_shift;
                d[y * RELOC_DESC_W + x] = val.clamp(0.0, 255.0) as u8;
            }
        }
        d
    }

    #[test]
    fn arroom_binary_roundtrip_preserves_planes_voxels_and_keyframes_compactly() {
        let mut map = SpatialMeshMap::default();
        map.ingest_ar_plane_polygons(
            &[
                RawArPlane {
                    raw_type: 0,
                    pos: Vec3::new(0.0, -1.2, -1.5),
                    rot: Quat::IDENTITY,
                    extent_x: 2.0,
                    extent_z: 2.0,
                    polygon_xz: vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                },
                RawArPlane {
                    raw_type: 0,
                    pos: Vec3::new(0.2, -0.45, -1.2),
                    rot: Quat::IDENTITY,
                    extent_x: 0.9,
                    extent_z: 0.6,
                    polygon_xz: vec![[-0.45, -0.3], [0.45, -0.3], [0.45, 0.3], [-0.45, 0.3]],
                },
            ],
            0.0,
        );
        let pts = [
            0.15, -0.80, -1.05, 0.9, 0.15, -0.80, -1.05, 0.9, 0.15, -0.80, -1.05, 0.9,
        ];
        map.ingest_point_cloud(3, &pts, Vec3::ZERO);
        map.step_greedy_meshing(4);

        let mut bank = KeyframeBank::default();
        assert!(bank.maybe_capture(Vec3::ZERO, Quat::IDENTITY, make_synthetic_desc(0.0, 0.0)));
        assert!(bank.maybe_capture(
            Vec3::new(0.8, 0.0, -0.4),
            Quat::from_rotation_y(0.8),
            make_synthetic_desc(1.4, 0.0)
        ));

        let bytes = serialize_room_arroom(&map, &bank);
        // Whole serialized room with 2 planes + voxel chunk + 2 keyframes stays tiny (< 3 KB!)
        assert!(bytes.len() < 3000);

        let loaded = deserialize_room_arroom(&bytes).expect("valid .arroom binary");
        assert_eq!(loaded.planes.len(), 2);
        assert!(loaded.tsdf_voxels_world.len() >= 2);
        assert_eq!(loaded.keyframes.len(), 2);
        assert!((loaded.floor_y.unwrap() - (-1.2)).abs() < 1e-4);
    }

    #[test]
    fn relocalization_4dof_aligns_rotated_session_under_lighting_shift() {
        let mut map_s1 = SpatialMeshMap::default();
        map_s1.ingest_ar_plane_polygons(
            &[
                RawArPlane {
                    raw_type: 0,
                    pos: Vec3::new(0.0, -1.2, -1.5),
                    rot: Quat::IDENTITY,
                    extent_x: 2.0,
                    extent_z: 2.0,
                    polygon_xz: vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                },
                RawArPlane {
                    raw_type: 0,
                    pos: Vec3::new(0.0, -0.45, -1.5),
                    rot: Quat::IDENTITY,
                    extent_x: 0.8,
                    extent_z: 0.6,
                    polygon_xz: vec![[-0.4, -0.3], [0.4, -0.3], [0.4, 0.3], [-0.4, 0.3]],
                },
            ],
            0.0,
        );
        let mut bank = KeyframeBank::default();
        let desc_anchor = make_synthetic_desc(0.7, 0.0);
        assert!(bank.maybe_capture(Vec3::ZERO, Quat::IDENTITY, desc_anchor));
        let bytes = serialize_room_arroom(&map_s1, &bank);

        // New ARCore session starts shifted (+1.5, +0.2, -0.8) and rotated +90 deg around Y,
        // with +35 brightness shift (lamp turned on)
        let mut loaded = deserialize_room_arroom(&bytes).unwrap();
        let curr_pos = Vec3::new(1.5, 0.2, -0.8);
        let curr_rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let live_desc = make_synthetic_desc(0.7, 35.0);

        // Frame 1 confirms candidate, Frame 2 locks auto-relocalization
        assert!(try_auto_relocalize(&mut loaded, curr_rot, &live_desc).is_none());
        let (kf_idx, score) = try_auto_relocalize(&mut loaded, curr_rot, &live_desc)
            .expect("should auto-relocalize on 2nd frame");
        assert_eq!(kf_idx, 0);
        assert!(score > 0.95);

        let kf = &loaded.keyframes[kf_idx];
        let (q_align, t_align) = compute_4dof_alignment(
            kf.cam_pos,
            kf.cam_rot,
            curr_pos,
            curr_rot,
            loaded.floor_y,
            None,
        );

        let mut map_s2 = SpatialMeshMap::default();
        map_s2.import_and_merge_transformed(
            &loaded.planes,
            &loaded.tsdf_voxels_world,
            q_align,
            t_align,
            curr_pos.y,
        );
        assert_eq!(map_s2.canonical_plane_count(), 2);

        // Table originally at (0, -0.45, -1.5) in Session 1 rotated +90 deg Y -> (-1.5, -0.45, 0) + (1.5, 0.2, -0.8) = (0.0, -0.25, -0.8)
        let table = map_s2
            .planes
            .iter()
            .find(|p| p.kind == SurfaceKind::Table)
            .expect("table restored");
        assert!((table.pos - Vec3::new(0.0, -0.25, -0.8)).length() < 0.03);
    }

    #[test]
    fn obj_exporter_emits_valid_wavefront_obj_faces() {
        let mut map = SpatialMeshMap::default();
        let dw = 64usize;
        let dh = 48usize;
        let mut depth = vec![0u8; dw * dh * 2];
        for i in 0..(dw * dh) {
            let le = 1400u16.to_le_bytes();
            depth[i * 2] = le[0];
            depth[i * 2 + 1] = le[1];
        }
        let conf = vec![190u8; dw * dh];
        for _ in 0..3 {
            map.ingest_depth_map(
                &depth,
                dw,
                dh,
                dw * 2,
                2,
                Some(&conf),
                dw,
                dh,
                dw,
                1,
                false,
                56.0,
                56.0,
                32.0,
                24.0,
                dw,
                dh,
                Vec3::ZERO,
                Quat::IDENTITY,
                true,
                1,
            );
        }
        let obj = export_room_to_obj(&mut map);
        assert!(obj.contains("g FlatSurfaces") || obj.contains("g DetailObjects"));
        assert!(obj.contains("v "));
        assert!(obj.contains("vn "));
        assert!(obj.contains("f 1//1 2//1 3//1"));
    }
}
