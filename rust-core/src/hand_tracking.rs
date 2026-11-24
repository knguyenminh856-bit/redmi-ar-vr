use crate::passthrough::{compile_program, PASSTHROUGH_DIST};
use glam::{Mat4, Vec2, Vec3};
use glow::HasContext;
use std::mem::size_of;

pub const MAX_HANDS: usize = 2;
pub const LANDMARKS_PER_HAND: usize = 21;
pub const FLOATS_PER_LANDMARK: usize = 5; // [ndc_x, ndc_y, wx, wy, wz]
const MAX_HAND_VERTICES: usize = 600;

const HAND_VERT: &str = r#"
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

const HAND_FRAG: &str = r#"
precision mediump float;
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HandGesture {
    None = 0,
    Fist = 1,
    Pinch = 2,
    FingerGun = 3,
    Peace = 4,
    OpenPalm = 5,
}

impl HandGesture {
    pub fn is_popup_action(self) -> bool {
        matches!(
            self,
            HandGesture::Fist | HandGesture::Pinch | HandGesture::FingerGun | HandGesture::Peace
        )
    }
}

#[derive(Clone, Debug)]
pub struct HandData {
    pub ndc: [Vec2; LANDMARKS_PER_HAND],
    pub world: [Vec3; LANDMARKS_PER_HAND],
    pub gesture: HandGesture,
}

impl HandData {
    pub fn from_packed_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() < LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK {
            return None;
        }
        let mut ndc = [Vec2::ZERO; LANDMARKS_PER_HAND];
        let mut world = [Vec3::ZERO; LANDMARKS_PER_HAND];
        for i in 0..LANDMARKS_PER_HAND {
            let base = i * FLOATS_PER_LANDMARK;
            ndc[i] = Vec2::new(slice[base], slice[base + 1]);
            world[i] = Vec3::new(slice[base + 2], slice[base + 3], slice[base + 4]);
        }
        let gesture = classify_gesture(&world);
        Some(Self { ndc, world, gesture })
    }
}

fn is_finger_extended(
    w: &[Vec3; LANDMARKS_PER_HAND],
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
    palm_len: f32,
) -> bool {
    let v1 = (w[pip] - w[mcp]).normalize_or_zero();
    let v2 = (w[dip] - w[pip]).normalize_or_zero();
    let v3 = (w[tip] - w[dip]).normalize_or_zero();
    v1.dot(v2) > 0.50
        && v2.dot(v3) > 0.30
        && (w[tip] - w[0]).length() > (w[pip] - w[0]).length() * 1.12
        && (w[tip] - w[mcp]).length() > palm_len * 0.60
}

fn is_finger_curled(
    w: &[Vec3; LANDMARKS_PER_HAND],
    mcp: usize,
    pip: usize,
    dip: usize,
    tip: usize,
    palm_len: f32,
) -> bool {
    let v1 = (w[pip] - w[mcp]).normalize_or_zero();
    let v2 = (w[dip] - w[pip]).normalize_or_zero();
    v1.dot(v2) < 0.42
        || (w[tip] - w[0]).length() < (w[pip] - w[0]).length() * 1.08
        || (w[tip] - w[mcp]).length() < palm_len * 0.68
}

/// Classifies static hand pose using scale-invariant 3D MediaPipe `worldLandmarks`.
pub fn classify_gesture(w: &[Vec3; LANDMARKS_PER_HAND]) -> HandGesture {
    let palm_len = (w[9] - w[0]).length().max(0.04);

    let index_ext = is_finger_extended(w, 5, 6, 7, 8, palm_len);
    let middle_ext = is_finger_extended(w, 9, 10, 11, 12, palm_len);
    let ring_ext = is_finger_extended(w, 13, 14, 15, 16, palm_len);
    let pinky_ext = is_finger_extended(w, 17, 18, 19, 20, palm_len);

    let index_curled = is_finger_curled(w, 5, 6, 7, 8, palm_len);
    let middle_curled = is_finger_curled(w, 9, 10, 11, 12, palm_len);
    let ring_curled = is_finger_curled(w, 13, 14, 15, 16, palm_len);
    let pinky_curled = is_finger_curled(w, 17, 18, 19, 20, palm_len);

    // 1. Pinch: thumb tip (4) touching index tip (8) while index is not tightly tucked in a fist.
    let pinch_dist = (w[4] - w[8]).length();
    if pinch_dist < palm_len * 0.36
        && ((w[8] - w[0]).length() > (w[5] - w[0]).length() * 1.12 || middle_ext || ring_ext)
    {
        return HandGesture::Pinch;
    }

    // 2. Finger Gun: thumb raised away from index knuckle & tip, index pointing (1- or 2-barrel), ring & pinky curled.
    let thumb_raised = (w[4] - w[5]).length() > palm_len * 0.40
        && (w[4] - w[8]).length() > palm_len * 0.46;
    let one_barrel = index_ext && middle_curled && ring_curled && pinky_curled;
    let two_barrel = index_ext
        && middle_ext
        && (w[8] - w[12]).length() < palm_len * 0.38
        && ring_curled
        && pinky_curled;
    if thumb_raised && (one_barrel || two_barrel) {
        return HandGesture::FingerGun;
    }

    // 3. Fist: all 4 main fingers closed.
    if !index_ext
        && !middle_ext
        && !ring_ext
        && !pinky_ext
        && index_curled
        && middle_curled
        && ring_curled
        && pinky_curled
    {
        return HandGesture::Fist;
    }

    // 4. Peace: index + middle split apart, ring + pinky curled.
    if index_ext
        && middle_ext
        && ring_curled
        && pinky_curled
        && (w[8] - w[12]).length() >= palm_len * 0.35
    {
        return HandGesture::Peace;
    }

    // 5. Open Palm: neutral reset gesture.
    if index_ext && middle_ext && ring_ext && pinky_ext {
        return HandGesture::OpenPalm;
    }

    HandGesture::None
}

/// Debounces gesture classification over consecutive frames so popups fire once per intentional pose.
#[derive(Debug, Default)]
pub struct GestureTracker {
    candidate: Option<HandGesture>,
    streak: u32,
    latched_popup_gesture: Option<HandGesture>,
    neutral_streak: u32,
    empty_frames: u32,
}

impl GestureTracker {
    pub fn update(&mut self, gesture: HandGesture, has_hands: bool) -> i32 {
        if !has_hands {
            self.candidate = None;
            self.streak = 0;
            self.empty_frames += 1;
            if self.empty_frames >= 3 {
                self.latched_popup_gesture = None;
                self.neutral_streak = 0;
            }
            return 0;
        }
        self.empty_frames = 0;

        if matches!(gesture, HandGesture::OpenPalm | HandGesture::None) {
            self.candidate = Some(gesture);
            self.streak = 0;
            self.neutral_streak += 1;
            if self.neutral_streak >= 2 {
                self.latched_popup_gesture = None;
            }
            return 0;
        }

        self.neutral_streak = 0;
        if self.candidate == Some(gesture) {
            self.streak += 1;
        } else {
            self.candidate = Some(gesture);
            self.streak = 1;
        }

        if gesture.is_popup_action()
            && self.streak >= 3
            && self.latched_popup_gesture != Some(gesture)
        {
            self.latched_popup_gesture = Some(gesture);
            return gesture as i32;
        }

        0
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HandVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

// 23 segments connecting Wrist (0) to all 5 finger chains + palm web (5-9-13-17)
const HAND_BONES: [(usize, usize, [f32; 4], f32); 23] = [
    // Thumb (gold)
    (0, 1, [1.0, 0.78, 0.22, 0.94], 0.011),
    (1, 2, [1.0, 0.78, 0.22, 0.94], 0.011),
    (2, 3, [1.0, 0.78, 0.22, 0.94], 0.011),
    (3, 4, [1.0, 0.78, 0.22, 0.94], 0.011),
    // Index (cyan)
    (0, 5, [0.0, 0.90, 1.00, 0.95], 0.011),
    (5, 6, [0.0, 0.90, 1.00, 0.95], 0.011),
    (6, 7, [0.0, 0.90, 1.00, 0.95], 0.011),
    (7, 8, [0.0, 0.90, 1.00, 0.95], 0.011),
    // Middle (spring green)
    (0, 9, [0.35, 0.96, 0.56, 0.94], 0.011),
    (9, 10, [0.35, 0.96, 0.56, 0.94], 0.011),
    (10, 11, [0.35, 0.96, 0.56, 0.94], 0.011),
    (11, 12, [0.35, 0.96, 0.56, 0.94], 0.011),
    // Ring (violet)
    (0, 13, [0.66, 0.48, 1.00, 0.94], 0.011),
    (13, 14, [0.66, 0.48, 1.00, 0.94], 0.011),
    (14, 15, [0.66, 0.48, 1.00, 0.94], 0.011),
    (15, 16, [0.66, 0.48, 1.00, 0.94], 0.011),
    // Pinky (coral)
    (0, 17, [1.00, 0.42, 0.58, 0.94], 0.011),
    (17, 18, [1.00, 0.42, 0.58, 0.94], 0.011),
    (18, 19, [1.00, 0.42, 0.58, 0.94], 0.011),
    (19, 20, [1.00, 0.42, 0.58, 0.94], 0.011),
    // Palm bridge (crisp white-cyan)
    (5, 9, [0.85, 0.95, 1.00, 0.78], 0.007),
    (9, 13, [0.85, 0.95, 1.00, 0.78], 0.007),
    (13, 17, [0.85, 0.95, 1.00, 0.78], 0.007),
];

fn finger_joint_color(idx: usize) -> [f32; 4] {
    match idx {
        0 => [1.0, 1.0, 1.0, 0.98],
        1..=4 => [1.0, 0.85, 0.35, 0.98],
        5..=8 => [0.25, 0.95, 1.0, 0.98],
        9..=12 => [0.45, 1.0, 0.65, 0.98],
        13..=16 => [0.76, 0.60, 1.0, 0.98],
        _ => [1.0, 0.55, 0.70, 0.98],
    }
}

fn ndc_to_quad(ndc: Vec2, half_w: f32, half_h: f32) -> Vec2 {
    Vec2::new(
        (ndc.x * half_w).clamp(-half_w, half_w),
        (ndc.y * half_h).clamp(-half_h, half_h),
    )
}

fn push_quad_ribbon(out: &mut Vec<HandVertex>, a: Vec2, b: Vec2, half_thick: f32, color: [f32; 4]) {
    let dir = b - a;
    let len = dir.length();
    if len < 1e-4 {
        return;
    }
    let perp = Vec2::new(-dir.y / len, dir.x / len) * half_thick;
    let z = -PASSTHROUGH_DIST;
    let p0 = [a.x - perp.x, a.y - perp.y, z];
    let p1 = [a.x + perp.x, a.y + perp.y, z];
    let p2 = [b.x + perp.x, b.y + perp.y, z];
    let p3 = [b.x - perp.x, b.y - perp.y, z];

    for p in [p0, p1, p2, p0, p2, p3] {
        out.push(HandVertex { pos: p, color });
    }
}

fn push_diamond(out: &mut Vec<HandVertex>, c: Vec2, radius: f32, color: [f32; 4]) {
    let z = -PASSTHROUGH_DIST;
    let top = [c.x, c.y + radius, z];
    let right = [c.x + radius, c.y, z];
    let bot = [c.x, c.y - radius, z];
    let left = [c.x - radius, c.y, z];
    for p in [top, left, bot, top, bot, right] {
        out.push(HandVertex { pos: p, color });
    }
}

/// Builds GPU triangle vertices for all active hands on the 1:1 passthrough quad plane.
pub fn build_hand_overlay_vertices(
    hands: &[HandData],
    half_w: f32,
    half_h: f32,
    out: &mut Vec<HandVertex>,
) {
    out.clear();
    for hand in hands.iter().take(MAX_HANDS) {
        let wrist = hand.ndc[0];
        if wrist.x.abs() > 1.35 || wrist.y.abs() > 1.35 {
            continue;
        }
        let mut pts = [Vec2::ZERO; LANDMARKS_PER_HAND];
        for i in 0..LANDMARKS_PER_HAND {
            pts[i] = ndc_to_quad(hand.ndc[i], half_w, half_h);
        }

        for &(a_idx, b_idx, color, half_thick) in &HAND_BONES {
            if out.len() + 6 > MAX_HAND_VERTICES {
                return;
            }
            push_quad_ribbon(out, pts[a_idx], pts[b_idx], half_thick, color);
        }

        for i in 0..LANDMARKS_PER_HAND {
            if out.len() + 6 > MAX_HAND_VERTICES {
                return;
            }
            let is_tip_or_wrist = matches!(i, 0 | 4 | 8 | 12 | 16 | 20);
            let r = if is_tip_or_wrist { 0.020 } else { 0.013 };
            push_diamond(out, pts[i], r, finger_joint_color(i));
        }

        if hand.gesture == HandGesture::Pinch && out.len() + 6 <= MAX_HAND_VERTICES {
            let pinch_mid = (pts[4] + pts[8]) * 0.5;
            push_diamond(out, pinch_mid, 0.032, [0.0, 1.0, 1.0, 0.95]);
        }
    }
}

pub struct HandOverlayRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    cpu_verts: Vec<HandVertex>,
    active_vertex_count: i32,
}

impl HandOverlayRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, HAND_VERT, HAND_FRAG)?;
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                (MAX_HAND_VERTICES * size_of::<HandVertex>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<HandVertex>() as i32;
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
                vao,
                vbo,
                u_mvp,
                cpu_verts: Vec::with_capacity(MAX_HAND_VERTICES),
                active_vertex_count: 0,
            })
        }
    }

    pub fn update_hands(&mut self, gl: &glow::Context, hands: &[HandData], half_w: f32, half_h: f32) {
        if hands.is_empty() {
            self.active_vertex_count = 0;
            return;
        }
        build_hand_overlay_vertices(hands, half_w, half_h, &mut self.cpu_verts);
        self.active_vertex_count = self.cpu_verts.len() as i32;
        if self.active_vertex_count == 0 {
            return;
        }
        unsafe {
            let bytes = std::slice::from_raw_parts(
                self.cpu_verts.as_ptr() as *const u8,
                self.cpu_verts.len() * size_of::<HandVertex>(),
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    pub fn draw(&self, gl: &glow::Context, mvp: &Mat4) {
        if self.active_vertex_count <= 0 {
            return;
        }
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, mvp.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.active_vertex_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
            gl.enable(glow::DEPTH_TEST);
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

    fn base_open_hand() -> [Vec3; LANDMARKS_PER_HAND] {
        let mut w = [Vec3::ZERO; LANDMARKS_PER_HAND];
        w[0] = Vec3::new(0.0, 0.0, 0.0);
        // Thumb extended out to the left/up
        w[1] = Vec3::new(-0.025, 0.02, 0.0);
        w[2] = Vec3::new(-0.045, 0.045, 0.0);
        w[3] = Vec3::new(-0.065, 0.070, 0.0);
        w[4] = Vec3::new(-0.085, 0.095, 0.0);

        // Straight vertical fingers from y=0.085..0.18
        let xs = [-0.032, -0.008, 0.016, 0.040];
        let bases = [5usize, 9, 13, 17];
        for (idx, &b) in bases.iter().enumerate() {
            let x = xs[idx];
            w[b] = Vec3::new(x, 0.085, 0.0);
            w[b + 1] = Vec3::new(x, 0.118, 0.0);
            w[b + 2] = Vec3::new(x, 0.146, 0.0);
            w[b + 3] = Vec3::new(x, 0.175, 0.0);
        }
        w
    }

    fn curl_finger(w: &mut [Vec3; LANDMARKS_PER_HAND], base: usize, x: f32) {
        w[base] = Vec3::new(x, 0.085, 0.0);
        w[base + 1] = Vec3::new(x, 0.105, -0.025);
        w[base + 2] = Vec3::new(x, 0.078, -0.045);
        w[base + 3] = Vec3::new(x, 0.058, -0.030);
    }

    #[test]
    fn classifies_open_palm_fist_pinch_gun_and_peace() {
        let open = base_open_hand();
        assert_eq!(classify_gesture(&open), HandGesture::OpenPalm);

        let mut fist = base_open_hand();
        w_curl_all(&mut fist);
        fist[4] = Vec3::new(-0.01, 0.08, -0.04);
        assert_eq!(classify_gesture(&fist), HandGesture::Fist);

        let mut pinch = base_open_hand();
        pinch[4] = Vec3::new(-0.032, 0.138, 0.0);
        pinch[8] = Vec3::new(-0.030, 0.142, 0.0);
        assert_eq!(classify_gesture(&pinch), HandGesture::Pinch);

        let mut gun = base_open_hand();
        curl_finger(&mut gun, 9, -0.008);
        curl_finger(&mut gun, 13, 0.016);
        curl_finger(&mut gun, 17, 0.040);
        // Point index along +X and thumb along +Y (L-shape gun)
        gun[5] = Vec3::new(0.01, 0.085, 0.0);
        gun[6] = Vec3::new(0.045, 0.085, 0.0);
        gun[7] = Vec3::new(0.075, 0.085, 0.0);
        gun[8] = Vec3::new(0.105, 0.085, 0.0);
        gun[4] = Vec3::new(-0.015, 0.155, 0.0);
        assert_eq!(classify_gesture(&gun), HandGesture::FingerGun);

        let mut peace = base_open_hand();
        peace[8] = Vec3::new(-0.055, 0.172, 0.0);
        peace[12] = Vec3::new(0.015, 0.174, 0.0);
        curl_finger(&mut peace, 13, 0.016);
        curl_finger(&mut peace, 17, 0.040);
        peace[4] = Vec3::new(0.0, 0.075, -0.03);
        assert_eq!(classify_gesture(&peace), HandGesture::Peace);
    }

    fn w_curl_all(w: &mut [Vec3; LANDMARKS_PER_HAND]) {
        curl_finger(w, 5, -0.032);
        curl_finger(w, 9, -0.008);
        curl_finger(w, 13, 0.016);
        curl_finger(w, 17, 0.040);
    }

    #[test]
    fn gesture_tracker_debounces_and_resets_on_open_palm() {
        let mut tracker = GestureTracker::default();
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 0);
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 0);
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 3);
        // Repeated latched frames should not spam popups
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 0);

        // Two neutral frames reset latch
        assert_eq!(tracker.update(HandGesture::OpenPalm, true), 0);
        assert_eq!(tracker.update(HandGesture::OpenPalm, true), 0);

        assert_eq!(tracker.update(HandGesture::FingerGun, true), 0);
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 0);
        assert_eq!(tracker.update(HandGesture::FingerGun, true), 3);
    }

    #[test]
    fn hand_overlay_builds_nonzero_skeleton_vertices() {
        let world = base_open_hand();
        let hand = HandData {
            ndc: [Vec2::new(0.0, 0.0); LANDMARKS_PER_HAND]
                .map(|_| Vec2::new(0.1, 0.1)),
            world,
            gesture: HandGesture::Pinch,
        };
        // Give non-degenerate points so ribbons have length > 1e-4
        let mut hand = hand;
        for i in 0..LANDMARKS_PER_HAND {
            hand.ndc[i] = Vec2::new(-0.3 + i as f32 * 0.03, -0.2 + (i % 5) as f32 * 0.12);
        }
        let mut verts = Vec::new();
        build_hand_overlay_vertices(&[hand], 1.5, 1.1, &mut verts);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 3, 0);
    }
}
