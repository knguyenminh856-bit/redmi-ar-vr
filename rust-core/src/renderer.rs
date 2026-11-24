use crate::gyro_predict::{predict_ar_rotation, sensor_to_head_quat};
use crate::hand_tracking::{
    GestureTracker, HandData, HandGesture, HandOverlayRenderer, FLOATS_PER_LANDMARK,
    LANDMARKS_PER_HAND, MAX_HANDS,
};
use crate::passthrough::{compute_quad_half_extents, PassthroughRenderer, PASSTHROUGH_DIST};
use crate::scene_3d::Scene3D;
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::time::Instant;

const HAND_TIMEOUT_MS: u128 = 350;

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

    pub hands: Vec<HandData>,
    pub last_hand_instant: Option<Instant>,
    pub gesture_tracker: GestureTracker,
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
            hands: Vec::with_capacity(MAX_HANDS),
            last_hand_instant: None,
            gesture_tracker: GestureTracker::default(),
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

    pub fn request_recenter(&mut self) {
        self.recenter_requested = true;
    }

    pub fn on_hand_landmarks(&mut self, hand_count: usize, packed: &[f32]) -> i32 {
        self.hands.clear();
        let count = hand_count.min(MAX_HANDS);
        let stride = LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK;
        for h in 0..count {
            let start = h * stride;
            let end = start + stride;
            if end <= packed.len() {
                if let Some(hand) = HandData::from_packed_slice(&packed[start..end]) {
                    self.hands.push(hand);
                }
            }
        }
        self.last_hand_instant = Some(Instant::now());

        let chosen_gesture = self
            .hands
            .iter()
            .find(|h| h.gesture.is_popup_action())
            .or_else(|| self.hands.first())
            .map(|h| h.gesture)
            .unwrap_or(HandGesture::None);

        self.gesture_tracker
            .update(chosen_gesture, !self.hands.is_empty())
    }

    /// Snapshots all values needed for this 120Hz frame and resolves the 6DoF/3DoF anchor origin.
    pub fn snapshot_frame(&mut self) -> FrameSnapshot {
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

        let active_hands = match self.last_hand_instant {
            Some(t) if t.elapsed().as_millis() <= HAND_TIMEOUT_MS => self.hands.clone(),
            _ => Vec::new(),
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
    pub hands: Vec<HandData>,
}

/// Computes a gravity-level anchor frame in front of the user's current horizontal gaze.
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

/// Computes split stereo eye view matrices for Left (-IPD/2) and Right (+IPD/2) eyes.
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
    width: i32,
    height: i32,
    start_time: Instant,
}

impl GlRenderer {
    pub fn new(gl: glow::Context) -> Result<Self, String> {
        let passthrough = PassthroughRenderer::new(&gl)?;
        let scene = Scene3D::new(&gl)?;
        let hand_overlay = HandOverlayRenderer::new(&gl)?;
        Ok(Self {
            gl,
            passthrough,
            scene,
            hand_overlay,
            width: 1200,
            height: 1080,
            start_time: Instant::now(),
        })
    }

    pub fn camera_texture_id(&self) -> u32 {
        self.passthrough.camera_texture_id()
    }

    pub fn resize(&mut self, width: i32, height: i32) {
        self.width = width.max(2);
        self.height = height.max(2);
    }

    pub fn draw_frame(&mut self, snap: &FrameSnapshot) {
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

        let proj_eye = Mat4::perspective_rh_gl(85.0_f32.to_radians(), eye_aspect, 0.05, 50.0);
        let (left_view, right_view) =
            compute_stereo_views(snap.head_pos_120hz, snap.head_rot_120hz, snap.ipd_meters);

        let t_cam_capture =
            Mat4::from_rotation_translation(snap.cam_rot_capture, snap.cam_pos_capture);

        let elapsed = self.start_time.elapsed().as_secs_f32();
        let (cube_model, octa_model) = Scene3D::object_transforms(&snap.anchor_origin, elapsed);

        unsafe {
            self.gl.viewport(0, 0, w, h);
            self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
            self.gl
                .clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);

            // Left eye viewport
            self.gl.viewport(0, 0, half_w, h);
            let left_vp = proj_eye * left_view;
            let left_pass_mvp = left_vp * t_cam_capture;
            self.passthrough
                .draw(&self.gl, &left_pass_mvp, snap.has_camera_frame);
            self.scene
                .draw(&self.gl, &left_vp, &cube_model, &octa_model);
            self.hand_overlay.draw(&self.gl, &left_pass_mvp);

            // Right eye viewport
            self.gl.viewport(half_w, 0, right_w, h);
            let right_vp = proj_eye * right_view;
            let right_pass_mvp = right_vp * t_cam_capture;
            self.passthrough
                .draw(&self.gl, &right_pass_mvp, snap.has_camera_frame);
            self.scene
                .draw(&self.gl, &right_vp, &cube_model, &octa_model);
            self.hand_overlay.draw(&self.gl, &right_pass_mvp);

            // Subtle 2px VR headset center ridge alignment guide
            self.gl.enable(glow::SCISSOR_TEST);
            self.gl.scissor((half_w - 1).max(0), 0, 2, h);
            self.gl.clear_color(0.12, 0.12, 0.15, 1.0);
            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.disable(glow::SCISSOR_TEST);
        }
    }

    pub fn destroy(&self) {
        self.passthrough.destroy(&self.gl);
        self.scene.destroy(&self.gl);
        self.hand_overlay.destroy(&self.gl);
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
