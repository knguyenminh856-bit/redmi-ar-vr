use crate::passthrough::{compile_program, PASSTHROUGH_DIST};
use glam::{Mat4, Vec2, Vec3};
use glow::HasContext;
use std::f32::consts::PI;
use std::mem::size_of;
use std::time::Instant;

pub const MAX_HANDS: usize = 2;
pub const LANDMARKS_PER_HAND: usize = 21;
pub const FLOATS_PER_LANDMARK: usize = 5; // [ndc_x, ndc_y, wx, wy, wz]
pub const FINGERTIP_INDICES: [usize; 5] = [4, 8, 12, 16, 20];

const MAX_HAND_VERTICES: usize = 780;
const MAX_MASK_VERTICES: usize = 420;
const COMBO_WINDOW_MS: u128 = 1650;

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
    ThumbsUp = 6,
    RockOn = 7,
    PointOne = 8,
}

#[derive(Copy, Clone, Debug)]
pub struct HandData {
    pub ndc: [Vec2; LANDMARKS_PER_HAND],
    pub world: [Vec3; LANDMARKS_PER_HAND],
    pub gesture: HandGesture,
    pub hand_depth_m: f32,
    pub tip_depths: [f32; 5],
}

impl HandData {
    pub fn from_packed_slice(slice: &[f32], tan_half_h: f32, tan_half_v: f32) -> Option<Self> {
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

        let safe_h = tan_half_h.clamp(0.2, 2.5);
        let safe_v = tan_half_v.clamp(0.2, 2.5);
        let d_tan = Vec2::new((ndc[9].x - ndc[0].x) * safe_h, (ndc[9].y - ndc[0].y) * safe_v).length();
        let w_xy = Vec2::new(world[9].x - world[0].x, world[9].y - world[0].y).length();
        let hand_depth_m = if d_tan > 0.015 && w_xy > 0.025 {
            (w_xy / d_tan).clamp(0.28, 0.90)
        } else {
            0.45
        };

        let mut tip_depths = [hand_depth_m; 5];
        for (slot, &tip_idx) in FINGERTIP_INDICES.iter().enumerate() {
            tip_depths[slot] = (hand_depth_m + (world[tip_idx].z - world[0].z)).clamp(0.25, 0.95);
        }

        Some(Self {
            ndc,
            world,
            gesture,
            hand_depth_m,
            tip_depths,
        })
    }
}

/// Adaptive low-latency 1-Euro Filter for jitter-free 120Hz fingertip smoothing.
#[derive(Copy, Clone, Debug)]
pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    initialized: bool,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32, d_cutoff: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            x_prev: 0.0,
            dx_prev: 0.0,
            initialized: false,
        }
    }

    pub fn reset(&mut self) {
        self.initialized = false;
        self.x_prev = 0.0;
        self.dx_prev = 0.0;
    }

    fn alpha(dt: f32, cutoff: f32) -> f32 {
        let tau = 1.0 / (2.0 * PI * cutoff.max(1e-3));
        1.0 / (1.0 + tau / dt.max(1e-4))
    }

    pub fn filter(&mut self, x: f32, dt: f32) -> f32 {
        if !self.initialized {
            self.initialized = true;
            self.x_prev = x;
            self.dx_prev = 0.0;
            return x;
        }
        let dt_safe = dt.clamp(1.0 / 240.0, 0.1);
        let dx = (x - self.x_prev) / dt_safe;
        let a_d = Self::alpha(dt_safe, self.d_cutoff);
        let dx_hat = self.dx_prev + a_d * (dx - self.dx_prev);
        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let a = Self::alpha(dt_safe, cutoff);
        let x_hat = self.x_prev + a * (x - self.x_prev);
        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        x_hat
    }
}

struct HandSlotFilters {
    ndc_x: [OneEuroFilter; LANDMARKS_PER_HAND],
    ndc_y: [OneEuroFilter; LANDMARKS_PER_HAND],
    hand_depth: OneEuroFilter,
    tip_depths: [OneEuroFilter; 5],
    active: bool,
}

impl HandSlotFilters {
    fn new() -> Self {
        let f = OneEuroFilter::new(1.5, 0.04, 1.0);
        Self {
            ndc_x: [f; LANDMARKS_PER_HAND],
            ndc_y: [f; LANDMARKS_PER_HAND],
            hand_depth: OneEuroFilter::new(1.0, 0.02, 1.0),
            tip_depths: [OneEuroFilter::new(1.2, 0.02, 1.0); 5],
            active: false,
        }
    }

    fn reset(&mut self) {
        if !self.active {
            return;
        }
        for i in 0..LANDMARKS_PER_HAND {
            self.ndc_x[i].reset();
            self.ndc_y[i].reset();
        }
        self.hand_depth.reset();
        for f in &mut self.tip_depths {
            f.reset();
        }
        self.active = false;
    }

    fn apply(&mut self, raw: &HandData, dt: f32) -> HandData {
        self.active = true;
        let mut out = *raw;
        for i in 0..LANDMARKS_PER_HAND {
            out.ndc[i].x = self.ndc_x[i].filter(raw.ndc[i].x, dt);
            out.ndc[i].y = self.ndc_y[i].filter(raw.ndc[i].y, dt);
        }
        out.hand_depth_m = self.hand_depth.filter(raw.hand_depth_m, dt);
        for slot in 0..5 {
            out.tip_depths[slot] = self.tip_depths[slot].filter(raw.tip_depths[slot], dt);
        }
        out
    }
}

pub struct HandSmoother {
    slots: [HandSlotFilters; MAX_HANDS],
    last_time: Option<Instant>,
}

impl Default for HandSmoother {
    fn default() -> Self {
        Self {
            slots: [HandSlotFilters::new(), HandSlotFilters::new()],
            last_time: None,
        }
    }
}

impl HandSmoother {
    pub fn smooth_all(
        &mut self,
        raw: &[Option<HandData>; MAX_HANDS],
        now: Instant,
    ) -> [Option<HandData>; MAX_HANDS] {
        let dt = self
            .last_time
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(1.0 / 60.0);
        self.last_time = Some(now);

        let mut out = [None; MAX_HANDS];
        for i in 0..MAX_HANDS {
            if let Some(ref h) = raw[i] {
                out[i] = Some(self.slots[i].apply(h, dt));
            } else {
                self.slots[i].reset();
            }
        }
        out
    }

    pub fn reset_all(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
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

/// Classifies 8 static hand gestures using scale-invariant 3D MediaPipe `worldLandmarks`.
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

    // 1. Pinch: thumb tip (4) touching index tip (8) while not tucked in a fist.
    let pinch_dist = (w[4] - w[8]).length();
    if pinch_dist < palm_len * 0.36
        && ((w[8] - w[0]).length() > (w[5] - w[0]).length() * 1.12 || middle_ext || ring_ext)
    {
        return HandGesture::Pinch;
    }

    // 2. Finger Gun vs PointOne (Index pointing).
    let thumb_raised_for_gun = (w[4] - w[5]).length() > palm_len * 0.44
        && (w[4] - w[10]).length() > palm_len * 0.48
        && (w[4] - w[8]).length() > palm_len * 0.48;
    let one_barrel = index_ext && middle_curled && ring_curled && pinky_curled;
    let two_barrel = index_ext
        && middle_ext
        && (w[8] - w[12]).length() < palm_len * 0.38
        && ring_curled
        && pinky_curled;
    if thumb_raised_for_gun && (one_barrel || two_barrel) {
        return HandGesture::FingerGun;
    }

    if one_barrel && !thumb_raised_for_gun {
        return HandGesture::PointOne;
    }

    // 3. RockOn: index + pinky extended, middle + ring curled.
    if index_ext && pinky_ext && middle_curled && ring_curled {
        return HandGesture::RockOn;
    }

    // 4. ThumbsUp vs Fist: all 4 fingers closed, distinguished by raised thumb.
    if !index_ext
        && !middle_ext
        && !ring_ext
        && !pinky_ext
        && index_curled
        && middle_curled
        && ring_curled
        && pinky_curled
    {
        let thumb_out = (w[4] - w[5]).length() > palm_len * 0.52
            && (w[4] - w[9]).length() > palm_len * 0.62;
        if thumb_out {
            return HandGesture::ThumbsUp;
        }
        return HandGesture::Fist;
    }

    // 5. Peace: index + middle split apart, ring + pinky curled.
    if index_ext
        && middle_ext
        && ring_curled
        && pinky_curled
        && (w[8] - w[12]).length() >= palm_len * 0.35
    {
        return HandGesture::Peace;
    }

    // 6. Open Palm: neutral reset / combo finisher.
    if index_ext && middle_ext && ring_ext && pinky_ext {
        return HandGesture::OpenPalm;
    }

    HandGesture::None
}

/// Debounces static poses and recognizes 2-step action combos within `COMBO_WINDOW_MS`.
/// Single popup codes: 1..=8 (except OpenPalm=5 which stays silent unless completing a combo).
/// Combo popup codes: 101..=105.
#[derive(Debug, Default)]
pub struct GestureTracker {
    candidate: Option<HandGesture>,
    streak: u32,
    latched_gesture: Option<HandGesture>,
    last_confirmed: Option<(HandGesture, Instant)>,
    none_streak: u32,
    empty_frames: u32,
}

impl GestureTracker {
    pub fn update_at(&mut self, gesture: HandGesture, has_hands: bool, now: Instant) -> i32 {
        if !has_hands {
            self.candidate = None;
            self.streak = 0;
            self.empty_frames += 1;
            if self.empty_frames >= 3 {
                self.latched_gesture = None;
            }
            if let Some((_, t)) = self.last_confirmed {
                if now.duration_since(t).as_millis() > COMBO_WINDOW_MS {
                    self.last_confirmed = None;
                }
            }
            return 0;
        }
        self.empty_frames = 0;

        if gesture == HandGesture::None {
            self.candidate = None;
            self.streak = 0;
            self.none_streak += 1;
            if self.none_streak >= 3 {
                self.latched_gesture = None;
            }
            return 0;
        }
        self.none_streak = 0;

        if self.candidate == Some(gesture) {
            self.streak += 1;
        } else {
            self.candidate = Some(gesture);
            self.streak = 1;
        }

        if self.streak >= 3 && self.latched_gesture != Some(gesture) {
            self.latched_gesture = Some(gesture);

            if let Some((prev, t)) = self.last_confirmed {
                if now.duration_since(t).as_millis() <= COMBO_WINDOW_MS {
                    let combo = match (prev, gesture) {
                        (HandGesture::Pinch, HandGesture::Fist) => 101,
                        (HandGesture::Fist, HandGesture::OpenPalm) => 102,
                        (HandGesture::Pinch, HandGesture::FingerGun) => 103,
                        (HandGesture::Peace, HandGesture::Fist) => 104,
                        (HandGesture::RockOn, HandGesture::OpenPalm) => 105,
                        _ => 0,
                    };
                    if combo > 0 {
                        self.last_confirmed = None;
                        return combo;
                    }
                }
            }

            self.last_confirmed = Some((gesture, now));
            if gesture != HandGesture::OpenPalm {
                return gesture as i32;
            }
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

const HAND_BONES: [(usize, usize, [f32; 4], f32); 23] = [
    // Thumb (gold)
    (0, 1, [1.0, 0.78, 0.22, 0.94], 0.010),
    (1, 2, [1.0, 0.78, 0.22, 0.94], 0.010),
    (2, 3, [1.0, 0.78, 0.22, 0.94], 0.010),
    (3, 4, [1.0, 0.78, 0.22, 0.94], 0.010),
    // Index (cyan)
    (0, 5, [0.0, 0.90, 1.00, 0.95], 0.010),
    (5, 6, [0.0, 0.90, 1.00, 0.95], 0.010),
    (6, 7, [0.0, 0.90, 1.00, 0.95], 0.010),
    (7, 8, [0.0, 0.90, 1.00, 0.95], 0.010),
    // Middle (spring green)
    (0, 9, [0.35, 0.96, 0.56, 0.94], 0.010),
    (9, 10, [0.35, 0.96, 0.56, 0.94], 0.010),
    (10, 11, [0.35, 0.96, 0.56, 0.94], 0.010),
    (11, 12, [0.35, 0.96, 0.56, 0.94], 0.010),
    // Ring (violet)
    (0, 13, [0.66, 0.48, 1.00, 0.94], 0.010),
    (13, 14, [0.66, 0.48, 1.00, 0.94], 0.010),
    (14, 15, [0.66, 0.48, 1.00, 0.94], 0.010),
    (15, 16, [0.66, 0.48, 1.00, 0.94], 0.010),
    // Pinky (coral)
    (0, 17, [1.00, 0.42, 0.58, 0.94], 0.010),
    (17, 18, [1.00, 0.42, 0.58, 0.94], 0.010),
    (18, 19, [1.00, 0.42, 0.58, 0.94], 0.010),
    (19, 20, [1.00, 0.42, 0.58, 0.94], 0.010),
    // Palm bridge
    (5, 9, [0.85, 0.95, 1.00, 0.78], 0.007),
    (9, 13, [0.85, 0.95, 1.00, 0.78], 0.007),
    (13, 17, [0.85, 0.95, 1.00, 0.78], 0.007),
];

const MASK_SEGMENTS: [(usize, usize); 20] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
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

fn push_quad_ribbon(out: &mut Vec<HandVertex>, a: Vec2, b: Vec2, z: f32, half_thick: f32, color: [f32; 4]) {
    let dir = b - a;
    let len = dir.length();
    if len < 1e-4 {
        return;
    }
    let perp = Vec2::new(-dir.y / len, dir.x / len) * half_thick;
    let p0 = [a.x - perp.x, a.y - perp.y, z];
    let p1 = [a.x + perp.x, a.y + perp.y, z];
    let p2 = [b.x + perp.x, b.y + perp.y, z];
    let p3 = [b.x - perp.x, b.y - perp.y, z];

    for p in [p0, p1, p2, p0, p2, p3] {
        out.push(HandVertex { pos: p, color });
    }
}

fn push_diamond(out: &mut Vec<HandVertex>, c: Vec2, z: f32, radius: f32, color: [f32; 4]) {
    let top = [c.x, c.y + radius, z];
    let right = [c.x + radius, c.y, z];
    let bot = [c.x, c.y - radius, z];
    let left = [c.x - radius, c.y, z];
    for p in [top, left, bot, top, bot, right] {
        out.push(HandVertex { pos: p, color });
    }
}

/// Builds GPU vertices for visible 5-finger bone ribbons + precision fingertip reticles.
pub fn build_hand_overlay_vertices(
    hands: &[Option<HandData>; MAX_HANDS],
    half_w: f32,
    half_h: f32,
    out: &mut Vec<HandVertex>,
) {
    out.clear();
    let z = -PASSTHROUGH_DIST;
    for hand in hands.iter().flatten() {
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
            push_quad_ribbon(out, pts[a_idx], pts[b_idx], z, half_thick, color);
        }

        for i in 0..LANDMARKS_PER_HAND {
            let is_tip = FINGERTIP_INDICES.contains(&i);
            if is_tip {
                if out.len() + 12 > MAX_HAND_VERTICES {
                    return;
                }
                // Precision dual-layer fingertip reticle: colored outer halo + sharp white center dot
                push_diamond(out, pts[i], z, 0.028, finger_joint_color(i));
                push_diamond(out, pts[i], z, 0.011, [1.0, 1.0, 1.0, 1.0]);
            } else {
                if out.len() + 6 > MAX_HAND_VERTICES {
                    return;
                }
                let r = if i == 0 { 0.022 } else { 0.012 };
                push_diamond(out, pts[i], z, r, finger_joint_color(i));
            }
        }

        if hand.gesture == HandGesture::Pinch && out.len() + 6 <= MAX_HAND_VERTICES {
            let pinch_mid = (pts[4] + pts[8]) * 0.5;
            push_diamond(out, pinch_mid, z, 0.036, [0.0, 1.0, 1.0, 0.96]);
        }
    }
}

/// Builds 3D camera-space hand silhouette triangles at estimated metric hand depth
/// for instant 120Hz hardware Z-buffer occlusion of virtual 3D objects.
pub fn build_hand_occluder_vertices(
    hands: &[Option<HandData>; MAX_HANDS],
    tan_half_h: f32,
    tan_half_v: f32,
    fov_scale: f32,
    out: &mut Vec<HandVertex>,
) {
    out.clear();
    let clamped_scale = fov_scale.clamp(0.45, 1.25);
    let safe_h = tan_half_h.clamp(0.2, 2.5) * clamped_scale;
    let safe_v = tan_half_v.clamp(0.2, 2.5) * clamped_scale;
    let mask_col = [1.0, 1.0, 1.0, 1.0];

    for hand in hands.iter().flatten() {
        let wrist = hand.ndc[0];
        if wrist.x.abs() > 1.35 || wrist.y.abs() > 1.35 {
            continue;
        }
        let d = hand.hand_depth_m.clamp(0.28, 0.90);
        let z = -d;
        let mut pts = [Vec2::ZERO; LANDMARKS_PER_HAND];
        for i in 0..LANDMARKS_PER_HAND {
            pts[i] = Vec2::new(hand.ndc[i].x * safe_h * d, hand.ndc[i].y * safe_v * d);
        }

        // Palm triangle fan: 0-1-5, 0-5-9, 0-9-13, 0-13-17
        let palm_tris = [(0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17)];
        for (a, b, c) in palm_tris {
            if out.len() + 3 > MAX_MASK_VERTICES {
                return;
            }
            for idx in [a, b, c] {
                out.push(HandVertex {
                    pos: [pts[idx].x, pts[idx].y, z],
                    color: mask_col,
                });
            }
        }

        // Thick finger capsule strips (~1.3cm physical half-width)
        let finger_half_w = 0.013;
        for &(a, b) in &MASK_SEGMENTS {
            if out.len() + 6 > MAX_MASK_VERTICES {
                return;
            }
            push_quad_ribbon(out, pts[a], pts[b], z, finger_half_w, mask_col);
        }
    }
}

pub struct HandOverlayRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    mask_vao: glow::VertexArray,
    mask_vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    cpu_verts: Vec<HandVertex>,
    mask_verts: Vec<HandVertex>,
    active_vertex_count: i32,
    active_mask_count: i32,
}

impl HandOverlayRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, HAND_VERT, HAND_FRAG)?;
            let (vao, vbo) = Self::create_vertex_stream(gl, program, MAX_HAND_VERTICES)?;
            let (mask_vao, mask_vbo) = Self::create_vertex_stream(gl, program, MAX_MASK_VERTICES)?;
            let u_mvp = gl.get_uniform_location(program, "u_mvp");

            Ok(Self {
                program,
                vao,
                vbo,
                mask_vao,
                mask_vbo,
                u_mvp,
                cpu_verts: Vec::with_capacity(MAX_HAND_VERTICES),
                mask_verts: Vec::with_capacity(MAX_MASK_VERTICES),
                active_vertex_count: 0,
                active_mask_count: 0,
            })
        }
    }

    unsafe fn create_vertex_stream(
        gl: &glow::Context,
        program: glow::Program,
        cap: usize,
    ) -> Result<(glow::VertexArray, glow::Buffer), String> {
        let vao = gl.create_vertex_array()?;
        let vbo = gl.create_buffer()?;

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_size(
            glow::ARRAY_BUFFER,
            (cap * size_of::<HandVertex>()) as i32,
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
        Ok((vao, vbo))
    }

    pub fn update_hands(
        &mut self,
        gl: &glow::Context,
        hands: &[Option<HandData>; MAX_HANDS],
        quad_hw: f32,
        quad_hh: f32,
        tan_half_h: f32,
        tan_half_v: f32,
        fov_scale: f32,
        build_occluder: bool,
    ) {
        if hands.iter().all(|h| h.is_none()) {
            self.active_vertex_count = 0;
            self.active_mask_count = 0;
            return;
        }

        build_hand_overlay_vertices(hands, quad_hw, quad_hh, &mut self.cpu_verts);
        self.active_vertex_count = self.cpu_verts.len() as i32;
        if self.active_vertex_count > 0 {
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

        if build_occluder {
            build_hand_occluder_vertices(
                hands,
                tan_half_h,
                tan_half_v,
                fov_scale,
                &mut self.mask_verts,
            );
            self.active_mask_count = self.mask_verts.len() as i32;
            if self.active_mask_count > 0 {
                unsafe {
                    let bytes = std::slice::from_raw_parts(
                        self.mask_verts.as_ptr() as *const u8,
                        self.mask_verts.len() * size_of::<HandVertex>(),
                    );
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.mask_vbo));
                    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                    gl.bind_buffer(glow::ARRAY_BUFFER, None);
                }
            }
        } else {
            self.active_mask_count = 0;
        }
    }

    /// Writes real hand geometry into the hardware depth buffer without modifying color pixels.
    pub fn draw_depth_occluder_mask(&self, gl: &glow::Context, pass_mvp: &Mat4) {
        if self.active_mask_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.color_mask(false, false, false, false);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.mask_vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, pass_mvp.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.active_mask_count);

            gl.bind_vertex_array(None);
            gl.color_mask(true, true, true, true);
        }
    }

    pub fn draw(&self, gl: &glow::Context, pass_mvp: &Mat4) {
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
                gl.uniform_matrix_4_f32_slice(Some(loc), false, pass_mvp.as_ref());
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
            gl.delete_buffer(self.mask_vbo);
            gl.delete_vertex_array(self.mask_vao);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn base_open_hand() -> [Vec3; LANDMARKS_PER_HAND] {
        let mut w = [Vec3::ZERO; LANDMARKS_PER_HAND];
        w[0] = Vec3::new(0.0, 0.0, 0.0);
        w[1] = Vec3::new(-0.025, 0.02, 0.0);
        w[2] = Vec3::new(-0.045, 0.045, 0.0);
        w[3] = Vec3::new(-0.065, 0.070, 0.0);
        w[4] = Vec3::new(-0.085, 0.095, 0.0);

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

    fn curl_all(w: &mut [Vec3; LANDMARKS_PER_HAND]) {
        curl_finger(w, 5, -0.032);
        curl_finger(w, 9, -0.008);
        curl_finger(w, 13, 0.016);
        curl_finger(w, 17, 0.040);
    }

    #[test]
    fn classifies_all_8_gestures() {
        let open = base_open_hand();
        assert_eq!(classify_gesture(&open), HandGesture::OpenPalm);

        let mut fist = base_open_hand();
        curl_all(&mut fist);
        fist[4] = Vec3::new(-0.01, 0.08, -0.04);
        assert_eq!(classify_gesture(&fist), HandGesture::Fist);

        let mut thumbs_up = base_open_hand();
        curl_all(&mut thumbs_up);
        thumbs_up[4] = Vec3::new(-0.08, 0.14, 0.0);
        assert_eq!(classify_gesture(&thumbs_up), HandGesture::ThumbsUp);

        let mut pinch = base_open_hand();
        pinch[4] = Vec3::new(-0.032, 0.138, 0.0);
        pinch[8] = Vec3::new(-0.030, 0.142, 0.0);
        assert_eq!(classify_gesture(&pinch), HandGesture::Pinch);

        let mut gun = base_open_hand();
        curl_finger(&mut gun, 9, -0.008);
        curl_finger(&mut gun, 13, 0.016);
        curl_finger(&mut gun, 17, 0.040);
        gun[5] = Vec3::new(0.01, 0.085, 0.0);
        gun[6] = Vec3::new(0.045, 0.085, 0.0);
        gun[7] = Vec3::new(0.075, 0.085, 0.0);
        gun[8] = Vec3::new(0.105, 0.085, 0.0);
        gun[4] = Vec3::new(-0.015, 0.155, 0.0);
        assert_eq!(classify_gesture(&gun), HandGesture::FingerGun);

        let mut point = base_open_hand();
        curl_finger(&mut point, 9, -0.008);
        curl_finger(&mut point, 13, 0.016);
        curl_finger(&mut point, 17, 0.040);
        point[4] = Vec3::new(-0.02, 0.085, -0.03);
        assert_eq!(classify_gesture(&point), HandGesture::PointOne);

        let mut peace = base_open_hand();
        peace[8] = Vec3::new(-0.055, 0.172, 0.0);
        peace[12] = Vec3::new(0.015, 0.174, 0.0);
        curl_finger(&mut peace, 13, 0.016);
        curl_finger(&mut peace, 17, 0.040);
        peace[4] = Vec3::new(0.0, 0.075, -0.03);
        assert_eq!(classify_gesture(&peace), HandGesture::Peace);

        let mut rock = base_open_hand();
        curl_finger(&mut rock, 9, -0.008);
        curl_finger(&mut rock, 13, 0.016);
        assert_eq!(classify_gesture(&rock), HandGesture::RockOn);
    }

    #[test]
    fn gesture_tracker_detects_action_combos_within_window() {
        let mut tracker = GestureTracker::default();
        let t0 = Instant::now();

        // Pinch -> Fist == 101
        assert_eq!(tracker.update_at(HandGesture::Pinch, true, t0), 0);
        assert_eq!(tracker.update_at(HandGesture::Pinch, true, t0), 0);
        assert_eq!(tracker.update_at(HandGesture::Pinch, true, t0), 2);
        let t1 = t0 + Duration::from_millis(300);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t1), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t1), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t1), 101);

        // Reset via OpenPalm, then Fist -> OpenPalm == 102
        let t2 = t1 + Duration::from_millis(2000);
        for _ in 0..3 {
            let _ = tracker.update_at(HandGesture::OpenPalm, true, t2);
        }
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t2), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t2), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t2), 1);
        let t3 = t2 + Duration::from_millis(400);
        assert_eq!(tracker.update_at(HandGesture::OpenPalm, true, t3), 0);
        assert_eq!(tracker.update_at(HandGesture::OpenPalm, true, t3), 0);
        assert_eq!(tracker.update_at(HandGesture::OpenPalm, true, t3), 102);

        // Peace -> Fist == 104
        let t4 = t3 + Duration::from_millis(200);
        assert_eq!(tracker.update_at(HandGesture::Peace, true, t4), 0);
        assert_eq!(tracker.update_at(HandGesture::Peace, true, t4), 0);
        assert_eq!(tracker.update_at(HandGesture::Peace, true, t4), 4);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t4), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t4), 0);
        assert_eq!(tracker.update_at(HandGesture::Fist, true, t4), 104);
    }

    #[test]
    fn one_euro_filter_smooths_step_without_divergence() {
        let mut filter = OneEuroFilter::new(1.5, 0.04, 1.0);
        assert!((filter.filter(0.0, 1.0 / 120.0) - 0.0).abs() < 1e-6);
        let mut last = 0.0;
        for _ in 0..30 {
            last = filter.filter(1.0, 1.0 / 120.0);
        }
        assert!(last > 0.7 && last <= 1.0);
    }

    #[test]
    fn overlay_and_occluder_build_valid_triangles() {
        let world = base_open_hand();
        let mut ndc = [Vec2::ZERO; LANDMARKS_PER_HAND];
        for i in 0..LANDMARKS_PER_HAND {
            ndc[i] = Vec2::new(-0.3 + i as f32 * 0.03, -0.2 + (i % 5) as f32 * 0.12);
        }
        let hand = HandData {
            ndc,
            world,
            gesture: HandGesture::Pinch,
            hand_depth_m: 0.45,
            tip_depths: [0.45; 5],
        };
        let hands = [Some(hand), None];

        let mut verts = Vec::new();
        build_hand_overlay_vertices(&hands, 1.5, 1.1, &mut verts);
        assert!(!verts.is_empty());
        assert_eq!(verts.len() % 3, 0);

        let mut mask = Vec::new();
        build_hand_occluder_vertices(&hands, 0.67, 0.50, 0.78, &mut mask);
        assert!(!mask.is_empty());
        assert_eq!(mask.len() % 3, 0);
    }
}
