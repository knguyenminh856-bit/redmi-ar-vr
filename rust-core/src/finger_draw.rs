use crate::depth_occlusion::DepthTextureGpu;
use crate::hand_tracking::{HandData, HandGesture, FINGERTIP_INDICES};
use crate::passthrough::compile_program;
use glam::{Mat4, Vec3};
use glow::HasContext;
use std::mem::size_of;

pub const MAX_DRAW_VERTICES: usize = 6000;
const MIN_STROKE_STEP_M: f32 = 0.006;
const MAX_STROKE_JUMP_M: f32 = 0.18;

const DRAW_VERT: &str = r#"
precision highp float;
attribute vec3 a_pos;
attribute vec4 a_color;
uniform mat4 u_mvp;
uniform mat4 u_inv_cam_capture;
varying vec4 v_color;
varying vec3 v_cam_local;
void main() {
    v_color = a_color;
    v_cam_local = (u_inv_cam_capture * vec4(a_pos, 1.0)).xyz;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const DRAW_FRAG: &str = r#"
precision mediump float;
uniform sampler2D u_depth_tex;
uniform float u_occlusion_enabled;
uniform vec2 u_tan_fov_scaled;
uniform vec2 u_d0;
uniform vec2 u_dx_vec;
uniform vec2 u_dy_vec;
varying vec4 v_color;
varying vec3 v_cam_local;
void main() {
    if (u_occlusion_enabled > 0.5) {
        float frag_d = -v_cam_local.z;
        if (frag_d > 0.05) {
            vec2 q_ndc = (v_cam_local.xy / frag_d) / u_tan_fov_scaled;
            if (abs(q_ndc.x) <= 1.0 && abs(q_ndc.y) <= 1.0) {
                vec2 st = q_ndc * 0.5 + 0.5;
                vec2 duv = u_d0 + st.x * u_dx_vec + st.y * u_dy_vec;
                vec2 rg = texture2D(u_depth_tex, duv).xy;
                float real_d = (rg.x * 255.0 + rg.y * 65280.0) * 0.001;
                if (real_d > 0.08 && real_d + 0.035 < frag_d) {
                    discard;
                }
            }
        }
    }
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

const FINGER_DRAW_COLORS: [[f32; 4]; 5] = [
    [1.00, 0.80, 0.20, 0.96], // Thumb: neon gold
    [0.05, 0.92, 1.00, 0.98], // Index: electric cyan
    [0.32, 0.98, 0.58, 0.96], // Middle: neon green
    [0.68, 0.45, 1.00, 0.96], // Ring: violet
    [1.00, 0.36, 0.64, 0.96], // Pinky: magenta-coral
];

pub struct FingerCanvas {
    prev_pts: [Option<Vec3>; 5],
    pub verts: Vec<DrawVertex>,
    pub upload_generation: u64,
}

impl Default for FingerCanvas {
    fn default() -> Self {
        Self {
            prev_pts: [None; 5],
            verts: Vec::with_capacity(MAX_DRAW_VERTICES),
            upload_generation: 0,
        }
    }
}

impl FingerCanvas {
    pub fn clear(&mut self) {
        self.prev_pts = [None; 5];
        if !self.verts.is_empty() {
            self.verts.clear();
            self.upload_generation = self.upload_generation.wrapping_add(1).max(1);
        }
    }

    pub fn break_strokes(&mut self) {
        self.prev_pts = [None; 5];
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
            self.prev_pts = [None; 5];
            return;
        }
        let Some(hand) = hand else {
            self.prev_pts = [None; 5];
            return;
        };

        let clamped_scale = fov_scale.clamp(0.45, 1.25);
        let safe_h = tan_half_h.clamp(0.2, 2.5) * clamped_scale;
        let safe_v = tan_half_v.clamp(0.2, 2.5) * clamped_scale;

        match mode {
            DrawMode::Off => {}
            DrawMode::SingleIndex => {
                if matches!(hand.gesture, HandGesture::PointOne | HandGesture::Pinch) {
                    let tip_idx = FINGERTIP_INDICES[1];
                    let ndc = hand.ndc[tip_idx];
                    if ndc.x.abs() <= 1.05 && ndc.y.abs() <= 1.05 {
                        let depth = hand.tip_depths[1].clamp(0.30, 0.80);
                        let p_cam = Vec3::new(ndc.x * safe_h * depth, ndc.y * safe_v * depth, -depth);
                        let p_world = t_cam_capture.transform_point3(p_cam);
                        self.push_slot_point(1, p_world, FINGER_DRAW_COLORS[1]);
                    } else {
                        self.prev_pts[1] = None;
                    }
                } else {
                    self.prev_pts[1] = None;
                }
            }
            DrawMode::FiveFingers => {
                if hand.gesture == HandGesture::Fist {
                    self.prev_pts = [None; 5];
                } else {
                    for slot in 0..5 {
                        let tip_idx = FINGERTIP_INDICES[slot];
                        let ndc = hand.ndc[tip_idx];
                        if ndc.x.abs() <= 1.05 && ndc.y.abs() <= 1.05 {
                            let depth = hand.tip_depths[slot].clamp(0.30, 0.80);
                            let p_cam =
                                Vec3::new(ndc.x * safe_h * depth, ndc.y * safe_v * depth, -depth);
                            let p_world = t_cam_capture.transform_point3(p_cam);
                            self.push_slot_point(slot, p_world, FINGER_DRAW_COLORS[slot]);
                        } else {
                            self.prev_pts[slot] = None;
                        }
                    }
                }
            }
        }
    }

    fn push_slot_point(&mut self, slot: usize, pt: Vec3, color: [f32; 4]) {
        if let Some(prev) = self.prev_pts[slot] {
            let dist = (pt - prev).length();
            if dist < MIN_STROKE_STEP_M {
                return;
            }
            if dist > MAX_STROKE_JUMP_M {
                self.prev_pts[slot] = Some(pt);
                return;
            }
            self.append_crossed_ribbon(prev, pt, 0.0045, color);
            self.prev_pts[slot] = Some(pt);
        } else {
            self.prev_pts[slot] = Some(pt);
        }
    }

    fn append_crossed_ribbon(&mut self, a: Vec3, b: Vec3, half_w: f32, color: [f32; 4]) {
        let dir = (b - a).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            return;
        }
        let ref_axis = if dir.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let side_a = dir.cross(ref_axis).normalize_or_zero() * half_w;
        let side_b = dir.cross(side_a).normalize_or_zero() * half_w;

        if self.verts.len() + 12 > MAX_DRAW_VERTICES {
            let drain_count = 1200.min(self.verts.len());
            self.verts.drain(0..drain_count);
        }

        for offset in [side_a, side_b] {
            let p0 = (a - offset).to_array();
            let p1 = (a + offset).to_array();
            let p2 = (b + offset).to_array();
            let p3 = (b - offset).to_array();
            for p in [p0, p1, p2, p0, p2, p3] {
                self.verts.push(DrawVertex { pos: p, color });
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
    u_inv_cam_capture: Option<glow::UniformLocation>,
    u_depth_tex: Option<glow::UniformLocation>,
    u_occlusion_enabled: Option<glow::UniformLocation>,
    u_tan_fov_scaled: Option<glow::UniformLocation>,
    u_d0: Option<glow::UniformLocation>,
    u_dx_vec: Option<glow::UniformLocation>,
    u_dy_vec: Option<glow::UniformLocation>,
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
                u_inv_cam_capture: gl.get_uniform_location(program, "u_inv_cam_capture"),
                u_depth_tex: gl.get_uniform_location(program, "u_depth_tex"),
                u_occlusion_enabled: gl.get_uniform_location(program, "u_occlusion_enabled"),
                u_tan_fov_scaled: gl.get_uniform_location(program, "u_tan_fov_scaled"),
                u_d0: gl.get_uniform_location(program, "u_d0"),
                u_dx_vec: gl.get_uniform_location(program, "u_dx_vec"),
                u_dy_vec: gl.get_uniform_location(program, "u_dy_vec"),
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

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gl: &glow::Context,
        view_proj: &Mat4,
        inv_cam_capture: &Mat4,
        depth_tex: &DepthTextureGpu,
        occlusion_enabled: bool,
        tan_half_h_scaled: f32,
        tan_half_v_scaled: f32,
    ) {
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
            if let Some(loc) = &self.u_inv_cam_capture {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, inv_cam_capture.as_ref());
            }

            let use_env_depth = occlusion_enabled && depth_tex.has_valid_depth;
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(depth_tex.texture));
            if let Some(loc) = &self.u_depth_tex {
                gl.uniform_1_i32(Some(loc), 1);
            }
            if let Some(loc) = &self.u_occlusion_enabled {
                gl.uniform_1_f32(Some(loc), if use_env_depth { 1.0 } else { 0.0 });
            }
            if let Some(loc) = &self.u_tan_fov_scaled {
                gl.uniform_2_f32(Some(loc), tan_half_h_scaled, tan_half_v_scaled);
            }
            if let Some(loc) = &self.u_d0 {
                gl.uniform_2_f32(Some(loc), depth_tex.d0[0], depth_tex.d0[1]);
            }
            if let Some(loc) = &self.u_dx_vec {
                gl.uniform_2_f32(Some(loc), depth_tex.dx[0], depth_tex.dx[1]);
            }
            if let Some(loc) = &self.u_dy_vec {
                gl.uniform_2_f32(Some(loc), depth_tex.dy[0], depth_tex.dy[1]);
            }

            gl.draw_arrays(glow::TRIANGLES, 0, self.active_vertex_count);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
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
            tip_depths: [0.45; 5],
        };
        for (slot, &idx) in FINGERTIP_INDICES.iter().enumerate() {
            h.ndc[idx] = Vec2::new(offset_x, (slot as f32 - 2.0) * 0.15);
        }
        h
    }

    #[test]
    fn single_and_five_finger_modes_generate_strokes_and_clear() {
        let mut canvas = FingerCanvas::default();
        let cam = Mat4::IDENTITY;

        let h1 = dummy_hand(HandGesture::PointOne, -0.2);
        let h2 = dummy_hand(HandGesture::PointOne, 0.0);
        canvas.update_strokes(DrawMode::SingleIndex, Some(&h1), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 0); // first point starts stroke
        canvas.update_strokes(DrawMode::SingleIndex, Some(&h2), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 12); // 1 crossed ribbon

        canvas.clear();
        assert_eq!(canvas.vertex_count(), 0);

        let f1 = dummy_hand(HandGesture::OpenPalm, -0.1);
        let f2 = dummy_hand(HandGesture::OpenPalm, 0.1);
        canvas.update_strokes(DrawMode::FiveFingers, Some(&f1), &cam, 0.67, 0.50, 1.0);
        canvas.update_strokes(DrawMode::FiveFingers, Some(&f2), &cam, 0.67, 0.50, 1.0);
        assert_eq!(canvas.vertex_count(), 60); // 5 fingers * 12 verts
    }
}
