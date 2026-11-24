use crate::finger_draw::{DrawMode, DrawVertex, FingerCanvas, FingerDrawRenderer};
use crate::gyro_predict::{predict_ar_rotation, sensor_to_head_quat};
use crate::hand_tracking::{
    smooth_hand_on_arrival, GestureTracker, HandData, HandGesture, HandOverlayRenderer,
    FLOATS_PER_LANDMARK, LANDMARKS_PER_HAND, MAX_HANDS,
};
use crate::passthrough::{compute_quad_half_extents, PassthroughRenderer, PASSTHROUGH_DIST};
use crate::physics_3d::{BodyShape, ImpactRing, PhysicsBody, PhysicsRenderer, PhysicsWorld};
use crate::room_persistence::{
    compute_4dof_alignment, deserialize_room_arroom, export_room_to_obj,
    pick_manual_anchor_keyframe, serialize_room_arroom, try_auto_relocalize, KeyframeBank,
    LoadedRoomMap,
};
use crate::room_scanner::{
    project_room_labels, AabbCollider, PlaneMarker, ProjectedLabel, ReticleHit,
    RoomScannerRenderer, SpatialMemoryCache, SurfaceKind, WorldObjectMarker,
};
use crate::scene_3d::Scene3D;
use crate::spatial_mesh::{RawArPlane, SpatialMeshMap, TriMeshRenderer, TriMeshVertex};
use crate::vision_ram::extract_latest_luma_descriptor;
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::time::Instant;

const HAND_TIMEOUT_MS: u128 = 350;

#[derive(Copy, Clone, Default)]
pub struct PerfMetrics {
    pub render_fps: f32,
    pub frame_ms: f32,
    pub active_hands: u32,
    pub current_gesture: i32,
    pub stroke_vertices: u32,
    pub index_tip_cam: [f32; 3],
    pub active_planes: u32,
    pub active_objects: u32,
    pub cached_planes: u32,
    pub voxel_count: u32,
    pub physics_bodies: u32,
    pub reticle_hit_kind: i32,
    pub reticle_surface_kind: i32,
    pub reticle_dist: f32,
    pub mesh_chunks: u32,
    pub mesh_triangles: u32,
    pub keyframe_count: u32,
    pub reloc_status: i32,
    pub reloc_best_pct: i32,
}

/// Thread-safe input state updated by Android sensor/camera/UI threads.
/// Contains zero OpenGL handles so `GlRenderer` stays exclusively on the GL thread.
pub struct SharedInputState {
    pub gyro_latest: Quat,
    pub gyro_at_ar_capture: Quat,
    pub has_gyro: bool,

    pub ar_pos: Vec3,
    pub ar_rot: Quat,
    pub has_ar_pose: bool,

    pub has_camera_frame: bool,
    pub cam_tan_half_h: f32,
    pub cam_tan_half_v: f32,
    pub cam_uvs: [f32; 8],

    pub fov_scale: f32,
    pub ipd_meters: f32,
    pub stereo_mode: bool,
    pub show_demo_scene: bool,
    pub recenter_requested: bool,
    pub anchor_origin: Option<Mat4>,
    pub anchored_floor_y: Option<f32>,

    pub hands: [Option<HandData>; MAX_HANDS],
    pub last_hand_instant: Option<Instant>,
    pub gesture_tracker: GestureTracker,
    pub current_gesture: HandGesture,

    pub draw_mode: DrawMode,
    pub finger_canvas: FingerCanvas,

    pub pull_closer: bool,
    pub boost_until: Option<Instant>,
    pub palette_idx: u32,

    pub room_scan_mode: i32,
    pub reticle: ReticleHit,
    pub spatial_cache: SpatialMemoryCache,
    pub spatial_mesh: SpatialMeshMap,
    pub cached_tri_mesh: Vec<TriMeshVertex>,
    pub cached_tri_src_gen: u64,
    pub cached_tri_upload_gen: u64,
    pub last_sent_tri_gen: u64,
    pub cached_tri_cull_pos: Vec3,
    pub cached_tri_chunks: u32,
    pub cached_tri_count: u32,
    pub keyframe_bank: KeyframeBank,
    pub pending_room: Option<LoadedRoomMap>,
    pub last_reloc_desc_seq: u64,
    pub reloc_status: i32,
    pub reloc_best_pct: i32,
    pub projected_labels: Vec<ProjectedLabel>,

    pub physics: PhysicsWorld,
    pub last_physics_step: Instant,

    pub perf: PerfMetrics,
}

impl Default for SharedInputState {
    fn default() -> Self {
        Self {
            gyro_latest: Quat::IDENTITY,
            gyro_at_ar_capture: Quat::IDENTITY,
            has_gyro: false,
            ar_pos: Vec3::ZERO,
            ar_rot: Quat::IDENTITY,
            has_ar_pose: false,
            has_camera_frame: false,
            cam_tan_half_h: 0.67,
            cam_tan_half_v: 0.50,
            cam_uvs: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            fov_scale: 1.0,
            ipd_meters: 0.064,
            stereo_mode: false,
            show_demo_scene: false,
            recenter_requested: false,
            anchor_origin: None,
            anchored_floor_y: None,
            hands: [None; MAX_HANDS],
            last_hand_instant: None,
            gesture_tracker: GestureTracker::default(),
            current_gesture: HandGesture::None,
            draw_mode: DrawMode::Off,
            finger_canvas: FingerCanvas::default(),
            pull_closer: false,
            boost_until: None,
            palette_idx: 0,
            room_scan_mode: 1,
            reticle: ReticleHit::default(),
            spatial_cache: SpatialMemoryCache::default(),
            spatial_mesh: SpatialMeshMap::default(),
            cached_tri_mesh: Vec::new(),
            cached_tri_src_gen: 0,
            cached_tri_upload_gen: 0,
            last_sent_tri_gen: 0,
            cached_tri_cull_pos: Vec3::ZERO,
            cached_tri_chunks: 0,
            cached_tri_count: 0,
            keyframe_bank: KeyframeBank::default(),
            pending_room: None,
            last_reloc_desc_seq: 0,
            reloc_status: 0,
            reloc_best_pct: 0,
            projected_labels: Vec::new(),
            physics: PhysicsWorld::default(),
            last_physics_step: Instant::now(),
            perf: PerfMetrics::default(),
        }
    }
}

impl SharedInputState {
    pub fn on_gyro_rotation(&mut self, qx: f32, qy: f32, qz: f32, qw: f32, display_rotation: i32) {
        let raw = Quat::from_xyzw(qx, qy, qz, qw);
        if raw.length_squared() < 1e-8 {
            return;
        }
        self.gyro_latest = sensor_to_head_quat(raw.normalize(), display_rotation);
        self.has_gyro = true;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_ar_frame(
        &mut self,
        px: f32,
        py: f32,
        pz: f32,
        rx: f32,
        ry: f32,
        rz: f32,
        rw: f32,
        tan_half_h: f32,
        tan_half_v: f32,
        uvs: [f32; 8],
        valid_tracking: bool,
    ) {
        self.has_camera_frame = true;
        self.cam_tan_half_h = tan_half_h;
        self.cam_tan_half_v = tan_half_v;
        self.cam_uvs = uvs;

        if valid_tracking {
            let q = Quat::from_xyzw(rx, ry, rz, rw);
            if q.length_squared() > 1e-8 {
                self.ar_pos = Vec3::new(px, py, pz);
                self.ar_rot = q.normalize();
                self.gyro_at_ar_capture = self.gyro_latest;
                self.has_ar_pose = true;
                if py.is_finite() {
                    self.anchored_floor_y.get_or_insert(py - 1.35);
                }
            }
        }
    }

    pub fn on_fallback_camera_frame(&mut self, tan_half_h: f32, tan_half_v: f32, uvs: [f32; 8]) {
        self.has_camera_frame = true;
        self.cam_tan_half_h = tan_half_h;
        self.cam_tan_half_v = tan_half_v;
        self.cam_uvs = uvs;
        self.ar_pos = Vec3::ZERO;
        self.ar_rot = self.gyro_latest;
        self.gyro_at_ar_capture = self.gyro_latest;
        self.has_ar_pose = false;
    }

    pub fn set_calibration(&mut self, fov_scale: f32, ipd_meters: f32) {
        self.fov_scale = fov_scale.clamp(0.45, 1.25);
        self.ipd_meters = ipd_meters.clamp(0.050, 0.078);
    }

    pub fn set_draw_mode(&mut self, draw_mode: i32) {
        let next_mode = DrawMode::from_i32(draw_mode);
        if next_mode != self.draw_mode {
            self.draw_mode = next_mode;
            self.finger_canvas.break_strokes();
        }
    }

    pub fn clear_drawings(&mut self) {
        self.finger_canvas.clear();
        self.perf.stroke_vertices = 0;
    }

    pub fn request_recenter(&mut self) {
        self.recenter_requested = true;
    }

    pub fn set_room_scan_mode(&mut self, mode: i32) {
        let clamped = mode.clamp(0, 4);
        self.room_scan_mode = clamped;
        if clamped == 0 {
            self.spatial_cache.clear();
            self.spatial_mesh.clear();
            self.keyframe_bank.clear();
            self.pending_room = None;
            self.reloc_status = 0;
            self.reloc_best_pct = 0;
            self.cached_tri_mesh.clear();
            self.cached_tri_src_gen = 0;
            self.cached_tri_upload_gen = self.cached_tri_upload_gen.wrapping_add(1);
            self.cached_tri_chunks = 0;
            self.cached_tri_count = 0;
            self.projected_labels.clear();
            self.reticle = ReticleHit::default();
            self.perf.active_planes = 0;
            self.perf.active_objects = 0;
            self.perf.cached_planes = 0;
            self.perf.voxel_count = 0;
            self.perf.mesh_chunks = 0;
            self.perf.mesh_triangles = 0;
            self.perf.keyframe_count = 0;
            self.perf.reloc_status = 0;
            self.perf.reloc_best_pct = 0;
            self.perf.reticle_hit_kind = 0;
            self.perf.reticle_surface_kind = 0;
            self.perf.reticle_dist = 0.0;
        } else if clamped != 2 {
            self.spatial_cache.clear_objects();
            self.perf.active_objects = 0;
        }
    }

    pub fn poll_visual_keyframes_and_reloc(&mut self) -> i32 {
        let Some((meta, desc)) = extract_latest_luma_descriptor(self.last_reloc_desc_seq) else {
            return self.reloc_status;
        };
        self.last_reloc_desc_seq = meta.seq;
        let (cam_pos, cam_rot) = self.current_cam_pose();

        if let Some(ref mut pending) = self.pending_room {
            if let Some((kf_idx, score)) = try_auto_relocalize(pending, cam_rot, &desc) {
                let pct = (score * 100.0).round().clamp(0.0, 100.0) as i32;
                self.apply_pending_room_alignment(kf_idx, cam_pos, cam_rot, true, pct);
                return 2;
            }
            self.reloc_best_pct = (pending.last_best_score * 100.0).round().clamp(0.0, 100.0) as i32;
            self.perf.reloc_best_pct = self.reloc_best_pct;
            return 1;
        }

        if self.room_scan_mode > 0 && self.has_ar_pose {
            self.keyframe_bank.maybe_capture(cam_pos, cam_rot, desc);
            self.perf.keyframe_count = self.keyframe_bank.len() as u32;
        }
        self.reloc_status
    }

    fn apply_pending_room_alignment(
        &mut self,
        kf_idx: usize,
        curr_pos: Vec3,
        curr_rot: Quat,
        is_auto: bool,
        pct: i32,
    ) -> bool {
        let Some(loaded) = self.pending_room.take() else {
            return false;
        };
        let (saved_pos, saved_rot) = loaded
            .keyframes
            .get(kf_idx)
            .map(|k| (k.cam_pos, k.cam_rot))
            .unwrap_or((Vec3::ZERO, Quat::IDENTITY));
        let (q_align, t_align) = compute_4dof_alignment(
            saved_pos,
            saved_rot,
            curr_pos,
            curr_rot,
            loaded.floor_y,
            self.spatial_mesh.inferred_floor_y(),
        );
        self.spatial_mesh.import_and_merge_transformed(
            &loaded.planes,
            &loaded.solid_voxels_world,
            q_align,
            t_align,
            curr_pos.y,
        );
        self.keyframe_bank
            .transform_and_merge(&loaded.keyframes, q_align, t_align);
        if let Some(fy) = self.spatial_mesh.inferred_floor_y() {
            self.anchored_floor_y = Some(fy);
        }
        self.reloc_status = if is_auto { 2 } else { 3 };
        self.reloc_best_pct = pct;
        self.perf.cached_planes = self.spatial_mesh.canonical_plane_count() as u32;
        self.perf.voxel_count = self.spatial_mesh.confirmed_voxel_count() as u32;
        self.perf.mesh_chunks = self.spatial_mesh.active_chunk_count() as u32;
        self.perf.keyframe_count = self.keyframe_bank.len() as u32;
        self.perf.reloc_status = self.reloc_status;
        self.perf.reloc_best_pct = self.reloc_best_pct;
        true
    }

    pub fn serialize_current_room(&self) -> Vec<u8> {
        serialize_room_arroom(&self.spatial_mesh, &self.keyframe_bank)
    }

    pub fn load_room_bytes(&mut self, bytes: &[u8]) -> Result<(usize, usize, usize), String> {
        let loaded = deserialize_room_arroom(bytes)?;
        let summary = (
            loaded.planes.len(),
            loaded.solid_voxels_world.len(),
            loaded.keyframes.len(),
        );
        self.pending_room = Some(loaded);
        self.reloc_status = 1;
        self.reloc_best_pct = 0;
        self.perf.reloc_status = 1;
        self.perf.reloc_best_pct = 0;
        Ok(summary)
    }

    pub fn force_relocalize_now(&mut self) -> bool {
        let (cam_pos, cam_rot) = self.current_cam_pose();
        let kf_idx = self
            .pending_room
            .as_ref()
            .and_then(|p| pick_manual_anchor_keyframe(p, cam_rot))
            .unwrap_or(0);
        self.apply_pending_room_alignment(kf_idx, cam_pos, cam_rot, false, 100)
    }

    pub fn cancel_pending_relocalize(&mut self) {
        self.pending_room = None;
        self.reloc_status = 0;
        self.reloc_best_pct = 0;
        self.perf.reloc_status = 0;
        self.perf.reloc_best_pct = 0;
    }

    pub fn export_current_room_obj(&mut self) -> String {
        export_room_to_obj(&mut self.spatial_mesh)
    }

    pub fn set_stereo_mode(&mut self, enabled: bool) {
        self.stereo_mode = enabled;
    }

    fn current_cam_pose(&self) -> (Vec3, Quat) {
        if self.has_ar_pose {
            (self.ar_pos, self.ar_rot)
        } else {
            let head = if self.has_gyro {
                self.gyro_latest
            } else {
                Quat::IDENTITY
            };
            (Vec3::ZERO, head)
        }
    }

    pub fn on_surface_reticle(
        &mut self,
        valid: bool,
        hit_kind: i32,
        pos: Vec3,
        normal: Vec3,
        distance: f32,
    ) -> (i32, i32, f32) {
        if self.room_scan_mode <= 0 {
            self.reticle = ReticleHit::default();
            self.perf.reticle_hit_kind = 0;
            self.perf.reticle_surface_kind = 0;
            self.perf.reticle_dist = 0.0;
            return (0, 0, 0.0);
        }

        let (cam_pos, cam_rot) = self.current_cam_pose();
        if valid && pos.is_finite() && normal.is_finite() {
            let n = {
                let cand = normal.normalize_or_zero();
                if cand.length_squared() > 1e-4 {
                    cand
                } else {
                    Vec3::Y
                }
            };
            // Rule 2: NEVER synthesize permanent patches during camera panning (eliminates floating phantom squares!)
            let skind = self.spatial_mesh.classify_surface_point(pos, n, cam_pos.y);
            let dist = if distance > 0.05 && distance.is_finite() {
                distance
            } else {
                (pos - cam_pos).length()
            };
            self.reticle = ReticleHit {
                valid: true,
                hit_kind,
                surface_kind: skind,
                pos,
                normal: n,
                distance: dist,
            };
            self.perf.reticle_hit_kind = hit_kind;
            self.perf.reticle_surface_kind = skind as i32;
            self.perf.reticle_dist = dist;
            return (hit_kind, skind as i32, dist);
        }

        // Ray-floor fallback when aiming downward below eye level
        let dir = (cam_rot * -Vec3::Z).normalize_or_zero();
        let floor_y = self
            .spatial_mesh
            .inferred_floor_y()
            .or_else(|| self.spatial_cache.inferred_floor_y())
            .unwrap_or_else(|| self.anchored_floor_y.unwrap_or(cam_pos.y - 1.35));
        if dir.y < -0.08 {
            let t = (floor_y - cam_pos.y) / dir.y;
            if (0.35..=5.0).contains(&t) {
                let hit_pos = cam_pos + dir * t;
                self.reticle = ReticleHit {
                    valid: true,
                    hit_kind: 3,
                    surface_kind: SurfaceKind::Floor,
                    pos: hit_pos,
                    normal: Vec3::Y,
                    distance: t,
                };
                self.perf.reticle_hit_kind = 3;
                self.perf.reticle_surface_kind = SurfaceKind::Floor as i32;
                self.perf.reticle_dist = t;
                return (3, SurfaceKind::Floor as i32, t);
            }
        }

        self.reticle = ReticleHit::default();
        self.perf.reticle_hit_kind = 0;
        self.perf.reticle_surface_kind = 0;
        self.perf.reticle_dist = 0.0;
        (0, 0, 0.0)
    }

    pub fn on_ar_planes(&mut self, count: usize, packed: &[f32]) {
        if self.room_scan_mode <= 0 || count == 0 {
            return;
        }
        let mut raw = Vec::with_capacity(count);
        for i in 0..count {
            let b = i * 10;
            if b + 10 > packed.len() {
                break;
            }
            raw.push(RawArPlane {
                raw_type: packed[b].round() as i32,
                pos: Vec3::new(packed[b + 1], packed[b + 2], packed[b + 3]),
                rot: Quat::from_xyzw(packed[b + 4], packed[b + 5], packed[b + 6], packed[b + 7]),
                extent_x: packed[b + 8].clamp(0.05, 8.0),
                extent_z: packed[b + 9].clamp(0.05, 8.0),
                polygon_xz: Vec::new(),
            });
        }
        self.on_ar_plane_polygons(raw);
    }

    pub fn on_ar_plane_polygons(&mut self, planes: Vec<RawArPlane>) {
        if self.room_scan_mode <= 0 {
            return;
        }
        self.spatial_mesh
            .ingest_ar_plane_polygons(&planes, self.ar_pos.y);
        self.perf.cached_planes = self.spatial_mesh.canonical_plane_count() as u32;
    }

    pub fn on_detected_objects(
        &mut self,
        count: usize,
        packed: &[f32],
        cam_pos: Vec3,
        cam_rot: Quat,
        tan_h: f32,
        tan_v: f32,
    ) {
        if self.room_scan_mode != 2 {
            self.spatial_cache.clear_objects();
            return;
        }
        self.spatial_cache.ingest_detected_objects(
            count,
            packed,
            cam_pos,
            cam_rot,
            tan_h,
            tan_v,
            self.room_scan_mode,
        );
    }

    pub fn set_projected_labels(&mut self, labels: Vec<ProjectedLabel>) {
        self.projected_labels = labels;
    }

    pub fn on_point_cloud(&mut self, count: usize, packed_xyzc: &[f32]) {
        if self.room_scan_mode <= 0 {
            return;
        }
        let cam_pos = if self.has_ar_pose { self.ar_pos } else { Vec3::ZERO };
        self.spatial_mesh
            .ingest_point_cloud(count, packed_xyzc, cam_pos);
        self.perf.voxel_count = self.spatial_mesh.confirmed_voxel_count() as u32;
    }

    pub fn spawn_physics_from_camera(&mut self, throw_speed: f32) {
        self.throw_physics_at_reticle(throw_speed);
    }

    pub fn drop_physics_at_reticle(&mut self) {
        let (cam_pos, cam_rot) = self.current_cam_pose();
        if self.reticle.valid && self.reticle.pos.is_finite() {
            // Only synthesize an explicit 36cm support patch when user explicitly drops onto a real Point normal
            if self.reticle.hit_kind == 2 {
                self.spatial_mesh.ensure_explicit_drop_patch(
                    self.reticle.pos,
                    self.reticle.normal,
                    cam_pos.y,
                );
            }
            self.physics
                .spawn_drop(self.reticle.pos, self.reticle.normal, None);
        } else {
            let dir = (cam_rot * -Vec3::Z).normalize_or_zero();
            let floor_y = self
                .spatial_mesh
                .inferred_floor_y()
                .unwrap_or_else(|| self.anchored_floor_y.unwrap_or(cam_pos.y - 1.35));
            let pos = Vec3::new(cam_pos.x + dir.x * 0.75, floor_y, cam_pos.z + dir.z * 0.75);
            self.physics.spawn_drop(pos, Vec3::Y, None);
        }
    }

    pub fn throw_physics_at_reticle(&mut self, throw_speed: f32) {
        let (cam_pos, cam_rot) = self.current_cam_pose();
        let dir = (cam_rot * -Vec3::Z).normalize_or_zero();
        if self.reticle.valid && self.reticle.pos.is_finite() {
            let dist = (self.reticle.pos - cam_pos).length();
            let launch_off = (dist * 0.30).clamp(0.14, 0.38);
            let origin = cam_pos + dir * launch_off + Vec3::new(0.0, 0.02, 0.0);
            self.physics
                .spawn_throw_at(origin, self.reticle.pos, dir, throw_speed, None);
        } else {
            let origin = cam_pos + dir * 0.35 + Vec3::new(0.0, 0.04, 0.0);
            self.physics.spawn_throw(origin, dir, throw_speed, None);
        }
    }

    pub fn clear_physics_bodies(&mut self) {
        self.physics.clear();
        self.perf.physics_bodies = 0;
    }

    pub fn on_hand_landmarks(&mut self, hand_count: usize, packed: &[f32]) -> i32 {
        let now = Instant::now();
        let prev_fresh = self
            .last_hand_instant
            .map(|t| now.duration_since(t).as_millis() <= HAND_TIMEOUT_MS)
            .unwrap_or(false);

        let mut next_hands = [None; MAX_HANDS];
        let count = hand_count.min(MAX_HANDS);
        let stride = LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK;
        let mut parsed_count = 0usize;

        for h in 0..count {
            let start = h * stride;
            let end = start + stride;
            if end <= packed.len() {
                if let Some(raw_hand) = HandData::from_packed_slice(
                    &packed[start..end],
                    self.cam_tan_half_h,
                    self.cam_tan_half_v,
                ) {
                    let prev_slot = if prev_fresh {
                        self.hands[parsed_count].as_ref()
                    } else {
                        None
                    };
                    next_hands[parsed_count] = Some(smooth_hand_on_arrival(prev_slot, raw_hand));
                    parsed_count += 1;
                }
            }
        }
        self.hands = next_hands;
        self.last_hand_instant = Some(now);

        let mut chosen = HandGesture::None;
        for slot in self.hands.iter().flatten() {
            if slot.gesture != HandGesture::None && slot.gesture != HandGesture::OpenPalm {
                chosen = slot.gesture;
                break;
            }
            if chosen == HandGesture::None {
                chosen = slot.gesture;
            }
        }
        self.current_gesture = chosen;

        // Update 3D fingertip drawing right when fresh MediaPipe landmarks arrive
        let (cam_pos_capture, cam_rot_capture) = if self.has_ar_pose {
            (self.ar_pos, self.ar_rot)
        } else {
            let head = if self.has_gyro {
                self.gyro_latest
            } else {
                Quat::IDENTITY
            };
            (Vec3::ZERO, head)
        };
        let t_cam_capture = Mat4::from_rotation_translation(cam_rot_capture, cam_pos_capture);
        self.finger_canvas.update_strokes(
            self.draw_mode,
            self.hands[0].as_ref(),
            &t_cam_capture,
            self.cam_tan_half_h,
            self.cam_tan_half_v,
            self.fov_scale,
        );

        let popup_code = self.gesture_tracker.update_at(chosen, parsed_count > 0, now);
        match popup_code {
            3 => {
                if let Some(h) = self.hands[0] {
                    let safe_h = self.cam_tan_half_h.clamp(0.2, 2.5);
                    let safe_v = self.cam_tan_half_v.clamp(0.2, 2.5);
                    let tip = h.ndc[8];
                    let origin_cam = Vec3::new(tip.x * safe_h * 0.35, tip.y * safe_v * 0.35, -0.35);
                    let origin = t_cam_capture.transform_point3(origin_cam);
                    let ray_cam = Vec3::new(tip.x * safe_h, tip.y * safe_v, -1.0);
                    let dir = (cam_rot_capture * ray_cam).normalize_or_zero();
                    self.physics
                        .spawn_throw(origin, dir, 2.5, Some(BodyShape::Sphere));
                }
            }
            101 => self.pull_closer = !self.pull_closer,
            102 => self.boost_until = Some(now + std::time::Duration::from_millis(3500)),
            103 => self.palette_idx = (self.palette_idx + 1) % 3,
            104 => self.finger_canvas.clear(),
            105 => self.recenter_requested = true,
            _ => {}
        }

        popup_code
    }

    pub fn record_render_timing(&mut self, fps: f32, frame_ms: f32) {
        self.perf.render_fps = fps;
        self.perf.frame_ms = frame_ms;
    }

    /// Snapshots all values needed for this 120Hz frame without redundant per-frame filtering.
    pub fn snapshot_frame(&mut self) -> FrameSnapshot {
        let now = Instant::now();
        let (cam_pos_capture, cam_rot_capture, head_rot_120hz) = if self.has_ar_pose {
            let predicted = if self.has_gyro {
                predict_ar_rotation(self.ar_rot, self.gyro_at_ar_capture, self.gyro_latest)
            } else {
                self.ar_rot
            };
            (self.ar_pos, self.ar_rot, predicted)
        } else {
            let head = if self.has_gyro {
                self.gyro_latest
            } else {
                Quat::IDENTITY
            };
            (Vec3::ZERO, head, head)
        };

        if self.recenter_requested || self.anchor_origin.is_none() {
            if self.has_ar_pose || self.has_gyro || self.recenter_requested {
                self.anchor_origin = Some(compute_yaw_aligned_anchor(cam_pos_capture, head_rot_120hz));
                self.recenter_requested = false;
            }
        }

        let hands_fresh = self
            .last_hand_instant
            .map(|t| now.duration_since(t).as_millis() <= HAND_TIMEOUT_MS)
            .unwrap_or(false);

        let active_hands = if hands_fresh {
            self.hands
        } else {
            self.current_gesture = HandGesture::None;
            self.finger_canvas.break_strokes();
            [None; MAX_HANDS]
        };

        let active_count = active_hands.iter().filter(|h| h.is_some()).count() as u32;
        self.perf.active_hands = active_count;
        self.perf.current_gesture = self.current_gesture as i32;
        self.perf.stroke_vertices = self.finger_canvas.vertex_count() as u32;
        let hand_world_pos = if let Some(h) = active_hands[0] {
            let safe_h = self.cam_tan_half_h.clamp(0.2, 2.5);
            let safe_v = self.cam_tan_half_v.clamp(0.2, 2.5);
            let d = h.hand_depth_m;
            let p_cam = Vec3::new(h.ndc[8].x * safe_h * d, h.ndc[8].y * safe_v * d, -d);
            self.perf.index_tip_cam = [p_cam.x, p_cam.y, d];
            Some(Mat4::from_rotation_translation(cam_rot_capture, cam_pos_capture).transform_point3(p_cam))
        } else {
            self.perf.index_tip_cam = [0.0, 0.0, 0.0];
            None
        };

        // Rule 3 & 4: FPS-governed budgeted incremental Greedy Meshing (1..2 dirty chunks per tick)
        let max_dirty = if self.perf.frame_ms > 10.5 { 1 } else { 2 };
        if self.room_scan_mode > 0 {
            self.spatial_mesh.step_greedy_meshing(max_dirty);
        }

        let active_planes = if self.room_scan_mode > 0 {
            self.spatial_mesh
                .snapshot_plane_markers(self.room_scan_mode == 2)
        } else {
            Vec::new()
        };
        let (_, active_objects) = if self.room_scan_mode == 2 {
            self.spatial_cache.snapshot_visible(2, now)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut blocks_25d = if self.room_scan_mode > 0 {
            self.spatial_mesh.collect_physics_chunk_aabbs(cam_pos_capture)
        } else {
            Vec::new()
        };
        if self.room_scan_mode == 2 {
            let obj_boxes = self.spatial_cache.build_25d_colliders(cam_pos_capture, 2);
            for b in obj_boxes.into_iter().filter(|b| b.is_object_box) {
                blocks_25d.push(b);
            }
        }

        // Rebuild Triangle Mesh VBO snapshot ONLY when in Mode 4 (🔺 Lưới △) AND map/camera actually changed
        let tri_mesh_mode = self.room_scan_mode == 4;
        let tri_mesh_verts = if tri_mesh_mode {
            let moved = (cam_pos_capture - self.cached_tri_cull_pos).length_squared() > 0.60 * 0.60;
            if self.cached_tri_src_gen != self.spatial_mesh.mesh_gen || moved || self.cached_tri_upload_gen == 0 {
                let cam_fwd = (head_rot_120hz * -Vec3::Z).normalize_or_zero();
                let (chunks, tris) = self.spatial_mesh.build_triangle_mesh_buffer(
                    cam_pos_capture,
                    cam_fwd,
                    &mut self.cached_tri_mesh,
                );
                self.cached_tri_src_gen = self.spatial_mesh.mesh_gen;
                self.cached_tri_cull_pos = cam_pos_capture;
                self.cached_tri_chunks = chunks;
                self.cached_tri_count = tris;
                self.cached_tri_upload_gen = self.cached_tri_upload_gen.wrapping_add(1).max(1);
            }
            if self.last_sent_tri_gen != self.cached_tri_upload_gen {
                self.last_sent_tri_gen = self.cached_tri_upload_gen;
                Some(self.cached_tri_mesh.clone())
            } else {
                None
            }
        } else {
            None
        };

        let dt_phys = now.duration_since(self.last_physics_step).as_secs_f32();
        self.last_physics_step = now;
        let fallback_floor_y = if let Some(inferred) = self.spatial_mesh.inferred_floor_y() {
            self.anchored_floor_y = Some(inferred);
            inferred
        } else {
            *self
                .anchored_floor_y
                .get_or_insert(cam_pos_capture.y - 1.35)
        };
        self.physics.step(
            dt_phys,
            &active_planes,
            &blocks_25d,
            fallback_floor_y,
            hand_world_pos,
        );

        self.perf.active_planes = active_planes.len() as u32;
        self.perf.active_objects = active_objects.len() as u32;
        self.perf.cached_planes = self.spatial_mesh.canonical_plane_count() as u32;
        self.perf.voxel_count = self.spatial_mesh.confirmed_voxel_count() as u32;
        self.perf.physics_bodies = self.physics.bodies.len() as u32;
        self.perf.mesh_chunks = self.spatial_mesh.active_chunk_count() as u32;
        self.perf.mesh_triangles = self.cached_tri_count;
        self.perf.keyframe_count = self.keyframe_bank.len() as u32;
        self.perf.reloc_status = self.reloc_status;
        self.perf.reloc_best_pct = self.reloc_best_pct;

        let boost_factor = match self.boost_until {
            Some(until) if until > now => {
                (until.duration_since(now).as_secs_f32() / 3.5).clamp(0.0, 1.0)
            }
            _ => {
                self.boost_until = None;
                0.0
            }
        };

        FrameSnapshot {
            cam_pos_capture,
            cam_rot_capture,
            head_pos_120hz: cam_pos_capture,
            head_rot_120hz,
            anchor_origin: self.anchor_origin.unwrap_or(Mat4::IDENTITY),
            has_camera_frame: self.has_camera_frame,
            tan_half_h: self.cam_tan_half_h,
            tan_half_v: self.cam_tan_half_v,
            uvs: self.cam_uvs,
            fov_scale: self.fov_scale,
            ipd_meters: self.ipd_meters,
            stereo_mode: self.stereo_mode,
            show_demo_scene: self.show_demo_scene,
            hands: active_hands,
            pull_closer: self.pull_closer,
            boost_factor,
            palette_idx: self.palette_idx,
            draw_upload_gen: self.finger_canvas.upload_generation,
            draw_verts: self.finger_canvas.verts.clone(),
            room_planes: active_planes,
            room_objects: active_objects,
            blocks_25d,
            reticle: if self.room_scan_mode > 0 {
                self.reticle
            } else {
                ReticleHit::default()
            },
            show_grid: matches!(self.room_scan_mode, 1 | 2),
            tri_mesh_mode,
            tri_mesh_gen: self.cached_tri_upload_gen,
            tri_mesh_verts,
            physics_bodies: self.physics.bodies.clone(),
            physics_rings: self.physics.impact_rings.clone(),
        }
    }
}

pub struct FrameSnapshot {
    pub cam_pos_capture: Vec3,
    pub cam_rot_capture: Quat,
    pub head_pos_120hz: Vec3,
    pub head_rot_120hz: Quat,
    pub anchor_origin: Mat4,
    pub has_camera_frame: bool,
    pub tan_half_h: f32,
    pub tan_half_v: f32,
    pub uvs: [f32; 8],
    pub fov_scale: f32,
    pub ipd_meters: f32,
    pub stereo_mode: bool,
    pub show_demo_scene: bool,
    pub hands: [Option<HandData>; MAX_HANDS],
    pub pull_closer: bool,
    pub boost_factor: f32,
    pub palette_idx: u32,
    pub draw_upload_gen: u64,
    pub draw_verts: Vec<DrawVertex>,
    pub room_planes: Vec<PlaneMarker>,
    pub room_objects: Vec<WorldObjectMarker>,
    pub blocks_25d: Vec<AabbCollider>,
    pub reticle: ReticleHit,
    pub show_grid: bool,
    pub tri_mesh_mode: bool,
    pub tri_mesh_gen: u64,
    pub tri_mesh_verts: Option<Vec<TriMeshVertex>>,
    pub physics_bodies: Vec<PhysicsBody>,
    pub physics_rings: Vec<ImpactRing>,
}

pub fn compute_yaw_aligned_anchor(head_pos: Vec3, head_rot: Quat) -> Mat4 {
    let raw_fwd = head_rot * -Vec3::Z;
    let horiz_fwd = Vec3::new(raw_fwd.x, 0.0, raw_fwd.z).normalize_or_zero();
    let anchor_rot = if horiz_fwd.length_squared() > 1e-4 {
        let right = horiz_fwd.cross(Vec3::Y).normalize();
        let basis = glam::Mat3::from_cols(right, Vec3::Y, -horiz_fwd);
        Quat::from_mat3(&basis)
    } else {
        Quat::IDENTITY
    };
    Mat4::from_rotation_translation(anchor_rot, head_pos)
}

pub fn compute_eye_projection(
    eye_aspect: f32,
    cam_tan_h: f32,
    cam_tan_v: f32,
    fov_scale: f32,
    overscan: f32,
) -> Mat4 {
    let safe_aspect = eye_aspect.max(0.2);
    let safe_h = cam_tan_h.clamp(0.2, 2.5);
    let safe_v = cam_tan_v.clamp(0.2, 2.5);
    let cover_tan_v = safe_v.min(safe_h / safe_aspect);
    let fill = fov_scale.clamp(0.50, 1.15) * overscan.clamp(1.00, 1.15);
    let eye_tan_v = (cover_tan_v / fill).clamp(0.18, 2.2);
    Mat4::perspective_rh_gl(2.0 * eye_tan_v.atan(), safe_aspect, 0.02, 50.0)
}

pub fn compute_quad_screen_bounds(
    pass_mvp: &Mat4,
    quad_hw: f32,
    quad_hh: f32,
    vp_x: i32,
    eye_w: i32,
    h: i32,
) -> ((i32, i32, i32, i32), (f32, f32, f32, f32)) {
    let corners = [
        Vec3::new(-quad_hw, -quad_hh, -PASSTHROUGH_DIST),
        Vec3::new(quad_hw, -quad_hh, -PASSTHROUGH_DIST),
        Vec3::new(-quad_hw, quad_hh, -PASSTHROUGH_DIST),
        Vec3::new(quad_hw, quad_hh, -PASSTHROUGH_DIST),
    ];
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    let mut any_valid = false;
    for c in corners {
        let clip = *pass_mvp * c.extend(1.0);
        if clip.w > 0.05 {
            let nx = clip.x / clip.w;
            let ny = clip.y / clip.w;
            min_x = min_x.min(nx);
            max_x = max_x.max(nx);
            min_y = min_y.min(ny);
            max_y = max_y.max(ny);
            any_valid = true;
        }
    }
    if !any_valid {
        return ((vp_x, 0, eye_w, h), (0.04, 0.96, 0.14, 0.90));
    }
    let nx0 = min_x.clamp(-1.0, 1.0);
    let nx1 = max_x.clamp(-1.0, 1.0);
    let ny0 = min_y.clamp(-1.0, 1.0);
    let ny1 = max_y.clamp(-1.0, 1.0);

    let sx0 = (vp_x + ((nx0 * 0.5 + 0.5) * eye_w as f32).floor() as i32).clamp(vp_x, vp_x + eye_w);
    let sx1 = (vp_x + ((nx1 * 0.5 + 0.5) * eye_w as f32).ceil() as i32).clamp(sx0, vp_x + eye_w);
    let sy0 = (((ny0 * 0.5 + 0.5) * h as f32).floor() as i32).clamp(0, h);
    let sy1 = (((ny1 * 0.5 + 0.5) * h as f32).ceil() as i32).clamp(sy0, h);

    let u_min = (nx0 * 0.5 + 0.5).clamp(0.0, 1.0);
    let u_max = (nx1 * 0.5 + 0.5).clamp(0.0, 1.0);
    let v_min = (0.5 - ny1 * 0.5).clamp(0.0, 1.0);
    let v_max = (0.5 - ny0 * 0.5).clamp(0.0, 1.0);
    (
        (sx0, sy0, (sx1 - sx0).max(1), (sy1 - sy0).max(1)),
        (u_min, u_max, v_min, v_max),
    )
}

pub fn compute_stereo_views(head_pos: Vec3, head_rot: Quat, ipd_meters: f32) -> (Mat4, Mat4) {
    let right_dir = head_rot * Vec3::X;
    let half_ipd = ipd_meters * 0.5;
    let left_pos = head_pos - right_dir * half_ipd;
    let right_pos = head_pos + right_dir * half_ipd;

    let left_view = Mat4::from_rotation_translation(head_rot, left_pos).inverse();
    let right_view = Mat4::from_rotation_translation(head_rot, right_pos).inverse();
    (left_view, right_view)
}

/// Owned exclusively by the Android GLSurfaceView GL thread.
pub struct GlRenderer {
    gl: glow::Context,
    passthrough: PassthroughRenderer,
    scene: Scene3D,
    hand_overlay: HandOverlayRenderer,
    finger_draw: FingerDrawRenderer,
    room_scanner: RoomScannerRenderer,
    tri_mesh: TriMeshRenderer,
    physics_renderer: PhysicsRenderer,
    pub gpu_renderer_name: String,
    width: i32,
    height: i32,
    start_time: Instant,
    last_frame_time: Option<Instant>,
    ema_frame_ms: f32,
}

impl GlRenderer {
    pub fn new(gl: glow::Context) -> Result<Self, String> {
        let gpu_renderer_name = unsafe { gl.get_parameter_string(glow::RENDERER) };
        let passthrough = PassthroughRenderer::new(&gl)?;
        let scene = Scene3D::new(&gl)?;
        let hand_overlay = HandOverlayRenderer::new(&gl)?;
        let finger_draw = FingerDrawRenderer::new(&gl)?;
        let room_scanner = RoomScannerRenderer::new(&gl)?;
        let tri_mesh = TriMeshRenderer::new(&gl)?;
        let physics_renderer = PhysicsRenderer::new(&gl)?;
        Ok(Self {
            gl,
            passthrough,
            scene,
            hand_overlay,
            finger_draw,
            room_scanner,
            tri_mesh,
            physics_renderer,
            gpu_renderer_name,
            width: 1200,
            height: 1080,
            start_time: Instant::now(),
            last_frame_time: None,
            ema_frame_ms: 8.33,
        })
    }

    pub fn camera_texture_id(&self) -> u32 {
        self.passthrough.camera_texture_id()
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width.max(2);
        self.height = height.max(2);
    }

    pub fn draw_frame(&mut self, snap: &FrameSnapshot) -> (f32, f32, Vec<ProjectedLabel>) {
        let now = Instant::now();
        if let Some(prev) = self.last_frame_time {
            let dt_ms = (now.duration_since(prev).as_secs_f32() * 1000.0).clamp(1.0, 200.0);
            self.ema_frame_ms = self.ema_frame_ms * 0.90 + dt_ms * 0.10;
        }
        self.last_frame_time = Some(now);
        let fps = if self.ema_frame_ms > 0.1 {
            1000.0 / self.ema_frame_ms
        } else {
            0.0
        };

        let w = self.width;
        let h = self.height;

        self.passthrough.update_quad(
            &self.gl,
            snap.tan_half_h,
            snap.tan_half_v,
            1.0,
            snap.uvs,
        );
        let (quad_hw, quad_hh) = compute_quad_half_extents(
            snap.tan_half_h,
            snap.tan_half_v,
            PASSTHROUGH_DIST,
            1.0,
        );
        self.hand_overlay
            .update_hands(&self.gl, &snap.hands, quad_hw, quad_hh);
        self.finger_draw
            .sync_if_dirty(&self.gl, &snap.draw_verts, snap.draw_upload_gen);
        self.room_scanner.update(
            &self.gl,
            &snap.room_planes,
            &snap.room_objects,
            &snap.blocks_25d,
            &snap.reticle,
            snap.show_grid,
            snap.head_pos_120hz,
            snap.head_rot_120hz,
        );
        if snap.tri_mesh_mode {
            if let Some(ref verts) = snap.tri_mesh_verts {
                self.tri_mesh
                    .sync_if_dirty(&self.gl, verts, snap.tri_mesh_gen);
            }
        }

        let t_cam_capture =
            Mat4::from_rotation_translation(snap.cam_rot_capture, snap.cam_pos_capture);
        let elapsed = self.start_time.elapsed().as_secs_f32();
        let (cube_model, octa_model) = if snap.show_demo_scene {
            Scene3D::object_transforms(
                &snap.anchor_origin,
                elapsed,
                snap.pull_closer,
                snap.boost_factor,
            )
        } else {
            (Mat4::IDENTITY, Mat4::IDENTITY)
        };

        let (left_vp, right_vp, left_uv_bounds, right_uv_bounds) = if !snap.stereo_mode {
            let eye_aspect = w as f32 / h as f32;
            let proj = compute_eye_projection(
                eye_aspect,
                snap.tan_half_h,
                snap.tan_half_v,
                snap.fov_scale,
                1.02,
            );
            let mono_view =
                Mat4::from_rotation_translation(snap.head_rot_120hz, snap.head_pos_120hz).inverse();
            let vp = proj * mono_view;
            let pass_mvp = vp * t_cam_capture;
            let (mono_scissor, mono_uv_bounds) =
                compute_quad_screen_bounds(&pass_mvp, quad_hw, quad_hh, 0, w, h);

            unsafe {
                self.gl.viewport(0, 0, w, h);
                self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                self.gl
                    .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

                self.passthrough
                    .draw(&self.gl, &pass_mvp, snap.has_camera_frame);
                if snap.show_demo_scene {
                    self.scene.draw(
                        &self.gl,
                        &vp,
                        &cube_model,
                        &octa_model,
                        snap.palette_idx,
                    );
                }
                self.physics_renderer.draw(
                    &self.gl,
                    &vp,
                    &snap.physics_bodies,
                    &snap.physics_rings,
                );
                self.gl.enable(glow::SCISSOR_TEST);
                self.gl.scissor(
                    mono_scissor.0,
                    mono_scissor.1,
                    mono_scissor.2,
                    mono_scissor.3,
                );
                if snap.tri_mesh_mode {
                    self.tri_mesh.draw(&self.gl, &vp);
                }
                self.room_scanner.draw_world_markers(&self.gl, &vp);
                self.gl.disable(glow::SCISSOR_TEST);
                self.finger_draw.draw(&self.gl, &vp);
                self.hand_overlay.draw(&self.gl, &pass_mvp);
            }

            (vp, vp, mono_uv_bounds, mono_uv_bounds)
        } else {
            let half_w = (w / 2).max(1);
            let right_w = (w - half_w).max(1);
            let eye_aspect = half_w as f32 / h as f32;

            let proj_eye = compute_eye_projection(
                eye_aspect,
                snap.tan_half_h,
                snap.tan_half_v,
                snap.fov_scale,
                1.06,
            );
            let (left_view, right_view) =
                compute_stereo_views(snap.head_pos_120hz, snap.head_rot_120hz, snap.ipd_meters);

            let left_vp = proj_eye * left_view;
            let left_pass_mvp = left_vp * t_cam_capture;
            let right_vp = proj_eye * right_view;
            let right_pass_mvp = right_vp * t_cam_capture;

            let (left_scissor, left_uv_bounds) =
                compute_quad_screen_bounds(&left_pass_mvp, quad_hw, quad_hh, 0, half_w, h);
            let (right_scissor, right_uv_bounds) =
                compute_quad_screen_bounds(&right_pass_mvp, quad_hw, quad_hh, half_w, right_w, h);

            unsafe {
                self.gl.viewport(0, 0, w, h);
                self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
                self.gl
                    .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

                // Left eye viewport
                self.gl.viewport(0, 0, half_w, h);
                self.passthrough
                    .draw(&self.gl, &left_pass_mvp, snap.has_camera_frame);
                if snap.show_demo_scene {
                    self.scene.draw(
                        &self.gl,
                        &left_vp,
                        &cube_model,
                        &octa_model,
                        snap.palette_idx,
                    );
                }
                self.physics_renderer.draw(
                    &self.gl,
                    &left_vp,
                    &snap.physics_bodies,
                    &snap.physics_rings,
                );
                self.gl.enable(glow::SCISSOR_TEST);
                self.gl.scissor(
                    left_scissor.0,
                    left_scissor.1,
                    left_scissor.2,
                    left_scissor.3,
                );
                if snap.tri_mesh_mode {
                    self.tri_mesh.draw(&self.gl, &left_vp);
                }
                self.room_scanner.draw_world_markers(&self.gl, &left_vp);
                self.gl.disable(glow::SCISSOR_TEST);
                self.finger_draw.draw(&self.gl, &left_vp);
                self.hand_overlay.draw(&self.gl, &left_pass_mvp);

                // Right eye viewport
                self.gl.viewport(half_w, 0, right_w, h);
                self.passthrough
                    .draw(&self.gl, &right_pass_mvp, snap.has_camera_frame);
                if snap.show_demo_scene {
                    self.scene.draw(
                        &self.gl,
                        &right_vp,
                        &cube_model,
                        &octa_model,
                        snap.palette_idx,
                    );
                }
                self.physics_renderer.draw(
                    &self.gl,
                    &right_vp,
                    &snap.physics_bodies,
                    &snap.physics_rings,
                );
                self.gl.enable(glow::SCISSOR_TEST);
                self.gl.scissor(
                    right_scissor.0,
                    right_scissor.1,
                    right_scissor.2,
                    right_scissor.3,
                );
                if snap.tri_mesh_mode {
                    self.tri_mesh.draw(&self.gl, &right_vp);
                }
                self.room_scanner.draw_world_markers(&self.gl, &right_vp);
                self.gl.disable(glow::SCISSOR_TEST);
                self.finger_draw.draw(&self.gl, &right_vp);
                self.hand_overlay.draw(&self.gl, &right_pass_mvp);

                // Subtle 2px VR headset center ridge alignment guide
                self.gl.enable(glow::SCISSOR_TEST);
                self.gl.scissor((half_w - 1).max(0), 0, 2, h);
                self.gl.clear_color(0.12, 0.12, 0.15, 1.0);
                self.gl.clear(glow::COLOR_BUFFER_BIT);
                self.gl.disable(glow::SCISSOR_TEST);
            }

            (left_vp, right_vp, left_uv_bounds, right_uv_bounds)
        };

        let labels = if snap.room_planes.is_empty() && snap.room_objects.is_empty() {
            Vec::new()
        } else {
            project_room_labels(
                &snap.room_planes,
                &snap.room_objects,
                &left_vp,
                &right_vp,
                left_uv_bounds,
                right_uv_bounds,
            )
        };

        (fps, self.ema_frame_ms, labels)
    }

    pub fn destroy(&self) {
        self.passthrough.destroy(&self.gl);
        self.scene.destroy(&self.gl);
        self.hand_overlay.destroy(&self.gl);
        self.finger_draw.destroy(&self.gl);
        self.room_scanner.destroy(&self.gl);
        self.tri_mesh.destroy(&self.gl);
        self.physics_renderer.destroy(&self.gl);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_views_offset_along_head_x_axis() {
        let head_pos = Vec3::new(1.0, 2.0, 3.0);
        let head_rot = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let (left_view, right_view) = compute_stereo_views(head_pos, head_rot, 0.064);

        let left_world = left_view.inverse().w_axis.truncate();
        let right_world = right_view.inverse().w_axis.truncate();
        assert!(((right_world - left_world).length() - 0.064).abs() < 1e-4);
    }

    #[test]
    fn passthrough_and_world_points_align_1_to_1_across_fov_scales() {
        let eye_aspect = 1200.0 / 1080.0;
        let tan_h = 0.65f32;
        let tan_v = 0.50f32;
        let (quad_hw, quad_hh) = compute_quad_half_extents(tan_h, tan_v, PASSTHROUGH_DIST, 1.0);

        let world_pt = Vec3::new(0.42, -0.18, -1.35);
        let cam_ndc_x = world_pt.x / (-world_pt.z * tan_h);
        let cam_ndc_y = world_pt.y / (-world_pt.z * tan_v);
        let quad_pt = Vec3::new(cam_ndc_x * quad_hw, cam_ndc_y * quad_hh, -PASSTHROUGH_DIST);

        for fov_scale in [0.65f32, 0.85, 1.00] {
            let proj = compute_eye_projection(eye_aspect, tan_h, tan_v, fov_scale, 1.06);
            let clip_world = proj * world_pt.extend(1.0);
            let clip_quad = proj * quad_pt.extend(1.0);
            let ndc_world = (clip_world.x / clip_world.w, clip_world.y / clip_world.w);
            let ndc_quad = (clip_quad.x / clip_quad.w, clip_quad.y / clip_quad.w);
            assert!((ndc_world.0 - ndc_quad.0).abs() < 1e-4);
            assert!((ndc_world.1 - ndc_quad.1).abs() < 1e-4);
        }

        // At 100% scale, camera quad corners must completely cover [-1, 1] NDC edge-to-edge (overscan >= 1.05 in stereo, >= 1.01 in mono)
        let full_proj = compute_eye_projection(eye_aspect, tan_h, tan_v, 1.0, 1.06);
        let clip_corner = full_proj * Vec3::new(quad_hw, quad_hh, -PASSTHROUGH_DIST).extend(1.0);
        assert!(clip_corner.x / clip_corner.w >= 1.05);
        assert!(clip_corner.y / clip_corner.w >= 1.05);

        let mono_proj = compute_eye_projection(2400.0 / 1080.0, tan_h, tan_v, 1.0, 1.02);
        let clip_mono = mono_proj * Vec3::new(quad_hw, quad_hh, -PASSTHROUGH_DIST).extend(1.0);
        assert!(clip_mono.x / clip_mono.w >= 1.01);
        assert!(clip_mono.y / clip_mono.w >= 1.01);
    }

    #[test]
    fn anchored_floor_locks_in_world_and_reticle_drop_does_not_chase_camera() {
        let mut state = SharedInputState::default();
        assert!(!state.stereo_mode);
        assert!(!state.show_demo_scene);
        assert_eq!(state.room_scan_mode, 1);

        // Camera starts at y = 0.0 -> world floor anchors at -1.35
        state.on_ar_frame(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.67,
            0.50,
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            true,
        );
        let snap1 = state.snapshot_frame();
        assert_eq!(state.anchored_floor_y, Some(-1.35));

        // Walk closer / crouch lower: anchored floor does NOT shift downward with camera!
        state.on_ar_frame(
            0.0,
            -0.45,
            -0.40,
            0.0,
            0.0,
            0.0,
            1.0,
            0.67,
            0.50,
            [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            true,
        );
        state.on_surface_reticle(true, 2, Vec3::new(0.0, -0.35, -0.75), Vec3::Y, 0.37);
        state.drop_physics_at_reticle();
        let snap2 = state.snapshot_frame();
        assert_eq!(state.anchored_floor_y, Some(-1.35));
        assert_eq!(snap2.physics_bodies.len(), 1);
        assert!(snap2.reticle.valid);
        assert_eq!(snap2.reticle.surface_kind, SurfaceKind::Table);
        let _ = snap1;
    }
}
