use crate::hand_tracking::{HandData, HandGesture, FINGERTIP_INDICES};
use crate::passthrough::compile_program;
use glam::{Mat4, Vec3};
use glow::HasContext;
use std::mem::size_of;

pub const MAX_DRAW_VERTICES: usize = 9600;
const DRAW_CANVAS_DIST: f32 = 0.65;
const MIN_STROKE_STEP_M: f32 = 0.005;
const MAX_STROKE_JUMP_M: f32 = 0.22;

const DRAW_VERT: &str = r#"
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

const DRAW_FRAG: &str = r#"
precision mediump float;
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DrawMode {
    Off = 0,
    SingleIndex = 1,
    FiveFingers = 2,
}

impl DrawMode {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => DrawMode::SingleIndex,
            2 => DrawMode::FiveFingers,
            _ => DrawMode::Off,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct DrawVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

const FINGER_CORE_COLORS: [[f32; 4]; 5] = [
    [1.00, 0.85, 0.28, 0.98], // Thumb: neon gold
    [0.12, 0.95, 1.00, 0.98], // Index: electric cyan
    [0.38, 1.00, 0.62, 0.98], // Middle: neon green
    [0.72, 0.52, 1.00, 0.98], // Ring: violet
    [1.00, 0.42, 0.68, 0.98], // Pinky: magenta-coral
];

pub struct FingerCanvas {
    last_raw: [Option<Vec3>; 5],
    last_mid: [Option<Vec3>; 5],
    pub verts: Vec<DrawVertex>,
    pub upload_generation: u64,
}

impl Default for FingerCanvas {
    fn default() -> Self {
        Self {
            last_raw: [None; 5],
            last_mid: [None; 5],
            verts: Vec::with_capacity(MAX_DRAW_VERTICES),
            upload_generation: 0,
        }
    }
}

impl FingerCanvas {
    pub fn clear(&mut self) {
        self.break_strokes();
        if !self.verts.is_empty() {
            self.verts.clear();
            self.upload_generation = self.upload_generation.wrapping_add(1).max(1);
        }
    }

    pub fn break_strokes(&mut self) {
        self.last_raw = [None; 5];
        self.last_mid = [None; 5];
    }

    pub fn vertex_count(&self) -> usize {
        self.verts.len()
    }

    pub fn update_strokes(
        &mut self,
        mode: DrawMode,
        hand: Option<&HandData>,
        t_cam_capture: &Mat4,
        tan_half_h: f32,
        tan_half_v: f32,
        fov_scale: f32,
    ) {
        if mode == DrawMode::Off {
            self.break_strokes();
            return;
        }
        let Some(hand) = hand else {
            self.break_strokes();
            return;
        };

        let clamped_scale = fov_scale.clamp(0.45, 1.25);
        let safe_h = tan_half_h.clamp(0.2, 2.5) * clamped_scale;
        let safe_v = tan_half_v.clamp(0.2, 2.5) * clamped_scale;

        match mode {
            DrawMode::Off => {}
            DrawMode::SingleIndex => {
                if matches!(
                    hand.gesture,
                    HandGesture::PointOne | HandGesture::Pinch | HandGesture::FingerGun
                ) {
                    let tip_idx = FINGERTIP_INDICES[1];
                    let ndc = hand.ndc[tip_idx];
                    if ndc.x.abs() <= 1.05 && ndc.y.abs() <= 1.05 {
                        let p_cam = Vec3::new(
                            ndc.x * safe_h * DRAW_CANVAS_DIST,
                            ndc.y * safe_v * DRAW_CANVAS_DIST,
                            -DRAW_CANVAS_DIST,
                        );
                        let p_world = t_cam_capture.transform_point3(p_cam);
                        self.push_slot_point(1, p_world, FINGER_CORE_COLORS[1]);
                    } else {
                        self.last_raw[1] = None;
                        self.last_mid[1] = None;
                    }
                } else {
                    self.last_raw[1] = None;
                    self.last_mid[1] = None;
                }
            }
            DrawMode::FiveFingers => {
                if hand.gesture == HandGesture::Fist {
                    self.break_strokes();
                } else {
                    for slot in 0..5 {
                        let tip_idx = FINGERTIP_INDICES[slot];
                        let ndc = hand.ndc[tip_idx];
                        if ndc.x.abs() <= 1.05 && ndc.y.abs() <= 1.05 {
                            let p_cam = Vec3::new(
                                ndc.x * safe_h * DRAW_CANVAS_DIST,
                                ndc.y * safe_v * DRAW_CANVAS_DIST,
                                -DRAW_CANVAS_DIST,
                            );
                            let p_world = t_cam_capture.transform_point3(p_cam);
                            self.push_slot_point(slot, p_world, FINGER_CORE_COLORS[slot]);
                        } else {
                            self.last_raw[slot] = None;
                            self.last_mid[slot] = None;
                        }
                    }
                }
            }
        }
    }

    fn push_slot_point(&mut self, slot: usize, pt: Vec3, core_color: [f32; 4]) {
        let Some(prev_raw) = self.last_raw[slot] else {
            self.last_raw[slot] = Some(pt);
            self.last_mid[slot] = Some(pt);
            return;
        };

        let dist = (pt - prev_raw).length();
        if dist < MIN_STROKE_STEP_M {
            return;
        }
        if dist > MAX_STROKE_JUMP_M {
            self.last_raw[slot] = Some(pt);
            self.last_mid[slot] = Some(pt);
            return;
        }

        let prev_mid = self.last_mid[slot].unwrap_or(prev_raw);
        let next_mid = (prev_raw + pt) * 0.5;
        let q_half = prev_mid * 0.25 + prev_raw * 0.5 + next_mid * 0.25;

        self.append_dual_ribbon(prev_mid, q_half, core_color);
        self.append_dual_ribbon(q_half, next_mid, core_color);

        self.last_raw[slot] = Some(pt);
        self.last_mid[slot] = Some(next_mid);
    }

    fn append_dual_ribbon(&mut self, a: Vec3, b: Vec3, core_color: [f32; 4]) {
        let dir = (b - a).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            return;
        }
        let ref_axis = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let u = dir.cross(ref_axis).normalize_or_zero();
        let v = dir.cross(u).normalize_or_zero();

        if self.verts.len() + 24 > MAX_DRAW_VERTICES {
            let drain_count = 1920.min(self.verts.len());
            self.verts.drain(0..drain_count);
        }

        let halo_color = [core_color[0], core_color[1], core_color[2], 0.36];
        for (half_w, color) in [(0.0080, halo_color), (0.0036, core_color)] {
            for offset in [u * half_w, v * half_w] {
                let p0 = (a - offset).to_array();
                let p1 = (a + offset).to_array();
                let p2 = (b + offset).to_array();
                let p3 = (b - offset).to_array();
                for p in [p0, p1, p2, p0, p2, p3] {
                    self.verts.push(DrawVertex { pos: p, color });
                }
            }
        }
        self.upload_generation = self.upload_generation.wrapping_add(1).max(1);
    }
}

pub struct FingerDrawRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    uploaded_gen: u64,
    active_vertex_count: i32,
}

impl FingerDrawRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, DRAW_VERT, DRAW_FRAG)?;
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;

            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_size(
                glow::ARRAY_BUFFER,
                (MAX_DRAW_VERTICES * size_of::<DrawVertex>()) as i32,
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<DrawVertex>() as i32;
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

            Ok(Self {
                program,
                vao,
                vbo,
                u_mvp: gl.get_uniform_location(program, "u_mvp"),
                uploaded_gen: 0,
                active_vertex_count: 0,
            })
        }
    }

    pub fn sync_if_dirty(&mut self, gl: &glow::Context, verts: &[DrawVertex], generation: u64) {
        if generation == self.uploaded_gen {
            return;
        }
        self.uploaded_gen = generation;
        let count = verts.len().min(MAX_DRAW_VERTICES);
        self.active_vertex_count = count as i32;
        if count == 0 {
            return;
        }
        unsafe {
            let bytes = std::slice::from_raw_parts(
                verts.as_ptr() as *const u8,
                count * size_of::<DrawVertex>(),
            );
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    pub fn draw(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.active_vertex_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));

            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }

            gl.draw_arrays(glow::TRIANGLES, 0, self.active_vertex_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
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
    use glam::Vec2;

    fn dummy_hand(gesture: HandGesture, offset_x: f32) -> HandData {
        let mut h = HandData {
            ndc: [Vec2::ZERO; 21],
            world: [Vec3::ZERO; 21],
            gesture,
            hand_depth_m: 0.45,
        };
        for (slot, &idx) in FINGERTIP_INDICES.iter().enumerate() {
            h.ndc[idx] = Vec2::new(offset_x, (slot as f32 - 2.0) * 0.15);
        }
        h
    }

    #[test]
    fn single_and_five_finger_modes_generate_dual_ribbons_and_clear() {
        let mut canvas = FingerCanvas::default();
        let cam = Mat4::IDENTITY;

        let h1 = dummy_hand(HandGesture::PointOne, -0.2);
        let h2 = dummy_hand(HandGesture::FingerGun, 0.0);
        canvas.update_strokes(DrawMode::SingleIndex, Some(&h1), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 0);
        canvas.update_strokes(DrawMode::SingleIndex, Some(&h2), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 48); // 2 Bezier sub-segments * 24 dual-ribbon verts

        canvas.clear();
        assert_eq!(canvas.vertex_count(), 0);

        let f1 = dummy_hand(HandGesture::OpenPalm, -0.1);
        let f2 = dummy_hand(HandGesture::OpenPalm, 0.1);
        canvas.update_strokes(DrawMode::FiveFingers, Some(&f1), &cam, 0.67, 0.50, 1.0);
        canvas.update_strokes(DrawMode::FiveFingers, Some(&f2), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 240); // 5 fingers * 48 verts
    }
}
