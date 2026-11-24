use crate::finger_draw::{DrawMode, DrawVertex, FingerCanvas, FingerDrawRenderer};
use crate::gyro_predict::{predict_ar_rotation, sensor_to_head_quat};
use crate::hand_tracking::{
    smooth_hand_on_arrival, GestureTracker, HandData, HandGesture, HandOverlayRenderer,
    FLOATS_PER_LANDMARK, LANDMARKS_PER_HAND, MAX_HANDS,
};
use crate::passthrough::{compute_quad_half_extents, PassthroughRenderer, PASSTHROUGH_DIST};
use crate::room_scanner::{
    parse_ar_planes, parse_detected_objects, project_room_labels, ObjectMarker, PlaneMarker,
    ProjectedLabel, RoomScannerRenderer, OBJECT_TIMEOUT_MS, PLANE_TIMEOUT_MS,
};
use crate::scene_3d::Scene3D;
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
    pub recenter_requested: bool,
    pub anchor_origin: Option<Mat4>,

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
    pub planes: Vec<PlaneMarker>,
    pub last_plane_instant: Option<Instant>,
    pub objects: Vec<ObjectMarker>,
    pub last_object_instant: Option<Instant>,
    pub projected_labels: Vec<ProjectedLabel>,

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
            fov_scale: 0.78,
            ipd_meters: 0.064,
            recenter_requested: false,
            anchor_origin: None,
            hands: [None; MAX_HANDS],
            last_hand_instant: None,
            gesture_tracker: GestureTracker::default(),
            current_gesture: HandGesture::None,
            draw_mode: DrawMode::Off,
            finger_canvas: FingerCanvas::default(),
            pull_closer: false,
            boost_until: None,
            palette_idx: 0,
            room_scan_mode: 2,
            planes: Vec::new(),
            last_plane_instant: None,
            objects: Vec::new(),
            last_object_instant: None,
            projected_labels: Vec::new(),
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
        let clamped = mode.clamp(0, 2);
        self.room_scan_mode = clamped;
        if clamped == 0 {
            self.planes.clear();
            self.objects.clear();
            self.projected_labels.clear();
            self.perf.active_planes = 0;
            self.perf.active_objects = 0;
        } else if clamped == 1 {
            self.objects.clear();
            self.perf.active_objects = 0;
        }
    }

    pub fn on_ar_planes(&mut self, count: usize, packed: &[f32]) {
        if self.room_scan_mode <= 0 {
            self.planes.clear();
            return;
        }
        self.planes = parse_ar_planes(count, packed, self.ar_pos.y, self.room_scan_mode);
        self.last_plane_instant = Some(Instant::now());
    }

    pub fn on_detected_objects(&mut self, count: usize, packed: &[f32]) {
        if self.room_scan_mode < 2 {
            self.objects.clear();
            return;
        }
        self.objects = parse_detected_objects(count, packed, self.room_scan_mode);
        self.last_object_instant = Some(Instant::now());
    }

    pub fn set_projected_labels(&mut self, labels: Vec<ProjectedLabel>) {
        self.projected_labels = labels;
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
        if let Some(h) = active_hands[0] {
            let safe_h = self.cam_tan_half_h.clamp(0.2, 2.5) * self.fov_scale.clamp(0.45, 1.25);
            let safe_v = self.cam_tan_half_v.clamp(0.2, 2.5) * self.fov_scale.clamp(0.45, 1.25);
            let d = h.hand_depth_m;
            self.perf.index_tip_cam = [h.ndc[8].x * safe_h * d, h.ndc[8].y * safe_v * d, d];
        } else {
            self.perf.index_tip_cam = [0.0, 0.0, 0.0];
        }

        let planes_fresh = self
            .last_plane_instant
            .map(|t| now.duration_since(t).as_millis() <= PLANE_TIMEOUT_MS)
            .unwrap_or(false);
        let active_planes = if self.room_scan_mode > 0 && planes_fresh {
            self.planes.clone()
        } else {
            if self.room_scan_mode > 0 && !planes_fresh {
                self.planes.clear();
            }
            Vec::new()
        };

        let objects_fresh = self
            .last_object_instant
            .map(|t| now.duration_since(t).as_millis() <= OBJECT_TIMEOUT_MS)
            .unwrap_or(false);
        let active_objects = if self.room_scan_mode >= 2 && objects_fresh {
            self.objects.clone()
        } else {
            if !objects_fresh {
                self.objects.clear();
            }
            Vec::new()
        };

        self.perf.active_planes = active_planes.len() as u32;
        self.perf.active_objects = active_objects.len() as u32;

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
            hands: active_hands,
            pull_closer: self.pull_closer,
            boost_factor,
            palette_idx: self.palette_idx,
            draw_upload_gen: self.finger_canvas.upload_generation,
            draw_verts: self.finger_canvas.verts.clone(),
            room_planes: active_planes,
            room_objects: active_objects,
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
    pub hands: [Option<HandData>; MAX_HANDS],
    pub pull_closer: bool,
    pub boost_factor: f32,
    pub palette_idx: u32,
    pub draw_upload_gen: u64,
    pub draw_verts: Vec<DrawVertex>,
    pub room_planes: Vec<PlaneMarker>,
    pub room_objects: Vec<ObjectMarker>,
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
        Ok(Self {
            gl,
            passthrough,
            scene,
            hand_overlay,
            finger_draw,
            room_scanner,
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
        let half_w = (w / 2).max(1);
        let right_w = (w - half_w).max(1);
        let eye_aspect = half_w as f32 / h as f32;

        self.passthrough.update_quad(
            &self.gl,
            snap.tan_half_h,
            snap.tan_half_v,
            snap.fov_scale,
            snap.uvs,
        );
        let (quad_hw, quad_hh) = compute_quad_half_extents(
            snap.tan_half_h,
            snap.tan_half_v,
            PASSTHROUGH_DIST,
            snap.fov_scale,
        );
        self.hand_overlay
            .update_hands(&self.gl, &snap.hands, quad_hw, quad_hh);
        self.finger_draw
            .sync_if_dirty(&self.gl, &snap.draw_verts, snap.draw_upload_gen);
        self.room_scanner.update(
            &self.gl,
            &snap.room_planes,
            &snap.room_objects,
            quad_hw,
            quad_hh,
        );

        let proj_eye = Mat4::perspective_rh_gl(85.0_f32.to_radians(), eye_aspect, 0.05, 50.0);
        let (left_view, right_view) =
            compute_stereo_views(snap.head_pos_120hz, snap.head_rot_120hz, snap.ipd_meters);

        let t_cam_capture =
            Mat4::from_rotation_translation(snap.cam_rot_capture, snap.cam_pos_capture);

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let (cube_model, octa_model) = Scene3D::object_transforms(
            &snap.anchor_origin,
            elapsed,
            snap.pull_closer,
            snap.boost_factor,
        );

        let left_vp = proj_eye * left_view;
        let left_pass_mvp = left_vp * t_cam_capture;
        let right_vp = proj_eye * right_view;
        let right_pass_mvp = right_vp * t_cam_capture;

        unsafe {
            self.gl.viewport(0, 0, w, h);
            self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            // Left eye viewport
            self.gl.viewport(0, 0, half_w, h);
            self.passthrough
                .draw(&self.gl, &left_pass_mvp, snap.has_camera_frame);
            self.scene.draw(
                &self.gl,
                &left_vp,
                &cube_model,
                &octa_model,
                snap.palette_idx,
            );
            self.room_scanner.draw_world_planes(&self.gl, &left_vp);
            self.finger_draw.draw(&self.gl, &left_vp);
            self.room_scanner
                .draw_object_badges(&self.gl, &left_pass_mvp);
            self.hand_overlay.draw(&self.gl, &left_pass_mvp);

            // Right eye viewport
            self.gl.viewport(half_w, 0, right_w, h);
            self.passthrough
                .draw(&self.gl, &right_pass_mvp, snap.has_camera_frame);
            self.scene.draw(
                &self.gl,
                &right_vp,
                &cube_model,
                &octa_model,
                snap.palette_idx,
            );
            self.room_scanner.draw_world_planes(&self.gl, &right_vp);
            self.finger_draw.draw(&self.gl, &right_vp);
            self.room_scanner
                .draw_object_badges(&self.gl, &right_pass_mvp);
            self.hand_overlay.draw(&self.gl, &right_pass_mvp);

            // Subtle 2px VR headset center ridge alignment guide
            self.gl.enable(glow::SCISSOR_TEST);
            self.gl.scissor((half_w - 1).max(0), 0, 2, h);
            self.gl.clear_color(0.12, 0.12, 0.15, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.disable(glow::SCISSOR_TEST);
        }

        let labels = if snap.room_planes.is_empty() && snap.room_objects.is_empty() {
            Vec::new()
        } else {
            project_room_labels(
                &snap.room_planes,
                &snap.room_objects,
                &t_cam_capture,
                quad_hw,
                quad_hh,
                &left_vp,
                &right_vp,
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
}
