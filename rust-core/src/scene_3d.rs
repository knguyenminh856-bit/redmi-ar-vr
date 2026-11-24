use crate::depth_occlusion::DepthTextureGpu;
use crate::passthrough::compile_program;
use glam::{Mat4, Vec3};
use glow::HasContext;
use std::mem::size_of;

const SCENE_VERT: &str = r#"
precision highp float;
attribute vec3 a_pos;
attribute vec3 a_normal;
attribute vec3 a_color;
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat4 u_inv_cam_capture;
varying vec3 v_normal_w;
varying vec3 v_color;
varying vec3 v_cam_local;
void main() {
    v_normal_w = normalize(mat3(u_model) * a_normal);
    v_color = a_color;
    v_cam_local = (u_inv_cam_capture * u_model * vec4(a_pos, 1.0)).xyz;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const SCENE_FRAG: &str = r#"
precision mediump float;
uniform sampler2D u_depth_tex;
uniform float u_occlusion_enabled;
uniform vec2 u_tan_fov_scaled;
uniform vec2 u_d0;
uniform vec2 u_dx_vec;
uniform vec2 u_dy_vec;
uniform float u_color_shift;
varying vec3 v_normal_w;
varying vec3 v_color;
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

    vec3 base_col = v_color;
    if (u_color_shift > 0.5 && u_color_shift < 1.5) {
        base_col = base_col.gbr;
    } else if (u_color_shift >= 1.5) {
        base_col = base_col.brg;
    }

    vec3 n = normalize(v_normal_w);
    vec3 light_dir = normalize(vec3(0.45, 0.80, 0.55));
    float diff = max(dot(n, light_dir), 0.0);
    float hemi = 0.35 + 0.20 * n.y;
    vec3 view_dir = normalize(vec3(0.0, 0.25, 1.0));
    vec3 half_vec = normalize(light_dir + view_dir);
    float spec = pow(max(dot(n, half_vec), 0.0), 24.0) * 0.28;
    vec3 lit = base_col * (hemi + diff * 0.72) + vec3(spec);
    gl_FragColor = vec4(lit, 1.0);
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex3D {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub fn build_cube_vertices(half_size: f32) -> Vec<Vertex3D> {
    let s = half_size;
    let faces: [([Vec3; 4], Vec3, [f32; 3]); 6] = [
        (
            [Vec3::new(-s, -s, s), Vec3::new(s, -s, s), Vec3::new(s, s, s), Vec3::new(-s, s, s)],
            Vec3::Z,
            [0.98, 0.36, 0.32],
        ),
        (
            [Vec3::new(s, -s, -s), Vec3::new(-s, -s, -s), Vec3::new(-s, s, -s), Vec3::new(s, s, -s)],
            -Vec3::Z,
            [0.14, 0.80, 0.72],
        ),
        (
            [Vec3::new(s, -s, s), Vec3::new(s, -s, -s), Vec3::new(s, s, -s), Vec3::new(s, s, s)],
            Vec3::X,
            [0.98, 0.72, 0.18],
        ),
        (
            [Vec3::new(-s, -s, -s), Vec3::new(-s, -s, s), Vec3::new(-s, s, s), Vec3::new(-s, s, -s)],
            -Vec3::X,
            [0.24, 0.62, 0.98],
        ),
        (
            [Vec3::new(-s, s, s), Vec3::new(s, s, s), Vec3::new(s, s, -s), Vec3::new(-s, s, -s)],
            Vec3::Y,
            [0.36, 0.92, 0.54],
        ),
        (
            [Vec3::new(-s, -s, -s), Vec3::new(s, -s, -s), Vec3::new(s, -s, s), Vec3::new(-s, -s, s)],
            -Vec3::Y,
            [0.64, 0.38, 0.96],
        ),
    ];

    let mut out = Vec::with_capacity(36);
    for (q, normal, color) in faces {
        let n = normal.to_array();
        for &idx in &[0usize, 1, 2, 0, 2, 3] {
            out.push(Vertex3D {
                pos: q[idx].to_array(),
                normal: n,
                color,
            });
        }
    }
    out
}

pub fn build_octahedron_vertices(radius: f32) -> Vec<Vertex3D> {
    let top = Vec3::new(0.0, radius * 1.25, 0.0);
    let bottom = Vec3::new(0.0, -radius * 1.25, 0.0);
    let ring = [
        Vec3::new(radius, 0.0, 0.0),
        Vec3::new(0.0, 0.0, -radius),
        Vec3::new(-radius, 0.0, 0.0),
        Vec3::new(0.0, 0.0, radius),
    ];

    let top_colors = [
        [0.18, 0.92, 0.96],
        [0.32, 0.78, 0.98],
        [0.14, 0.86, 0.88],
        [0.44, 0.94, 0.92],
    ];
    let bot_colors = [
        [0.98, 0.78, 0.24],
        [0.94, 0.58, 0.22],
        [0.98, 0.84, 0.34],
        [0.92, 0.64, 0.28],
    ];

    let mut out = Vec::with_capacity(24);
    for i in 0..4 {
        let a = ring[i];
        let b = ring[(i + 1) % 4];
        push_flat_tri(&mut out, a, b, top, top_colors[i]);
        push_flat_tri(&mut out, b, a, bottom, bot_colors[i]);
    }
    out
}

fn push_flat_tri(out: &mut Vec<Vertex3D>, v0: Vec3, v1: Vec3, v2: Vec3, color: [f32; 3]) {
    let n = (v1 - v0).cross(v2 - v0).normalize_or_zero().to_array();
    for v in [v0, v1, v2] {
        out.push(Vertex3D {
            pos: v.to_array(),
            normal: n,
            color,
        });
    }
}

struct MeshGpu {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vertex_count: i32,
}

impl MeshGpu {
    unsafe fn from_vertices(
        gl: &glow::Context,
        program: glow::Program,
        verts: &[Vertex3D],
    ) -> Result<Self, String> {
        let vao = gl.create_vertex_array()?;
        let vbo = gl.create_buffer()?;

        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let bytes =
            std::slice::from_raw_parts(verts.as_ptr() as *const u8, std::mem::size_of_val(verts));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);

        let stride = size_of::<Vertex3D>() as i32;
        if let Some(a_pos) = gl.get_attrib_location(program, "a_pos") {
            gl.enable_vertex_attrib_array(a_pos);
            gl.vertex_attrib_pointer_f32(a_pos, 3, glow::FLOAT, false, stride, 0);
        }
        if let Some(a_normal) = gl.get_attrib_location(program, "a_normal") {
            gl.enable_vertex_attrib_array(a_normal);
            gl.vertex_attrib_pointer_f32(
                a_normal,
                3,
                glow::FLOAT,
                false,
                stride,
                (3 * size_of::<f32>()) as i32,
            );
        }
        if let Some(a_color) = gl.get_attrib_location(program, "a_color") {
            gl.enable_vertex_attrib_array(a_color);
            gl.vertex_attrib_pointer_f32(
                a_color,
                3,
                glow::FLOAT,
                false,
                stride,
                (6 * size_of::<f32>()) as i32,
            );
        }

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);

        Ok(Self {
            vao,
            vbo,
            vertex_count: verts.len() as i32,
        })
    }

    unsafe fn destroy(&self, gl: &glow::Context) {
        gl.delete_buffer(self.vbo);
        gl.delete_vertex_array(self.vao);
    }
}

pub struct Scene3D {
    program: glow::Program,
    cube: MeshGpu,
    octahedron: MeshGpu,
    u_mvp: Option<glow::UniformLocation>,
    u_model: Option<glow::UniformLocation>,
    u_inv_cam_capture: Option<glow::UniformLocation>,
    u_depth_tex: Option<glow::UniformLocation>,
    u_occlusion_enabled: Option<glow::UniformLocation>,
    u_tan_fov_scaled: Option<glow::UniformLocation>,
    u_d0: Option<glow::UniformLocation>,
    u_dx_vec: Option<glow::UniformLocation>,
    u_dy_vec: Option<glow::UniformLocation>,
    u_color_shift: Option<glow::UniformLocation>,
}

impl Scene3D {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, SCENE_VERT, SCENE_FRAG)?;
            let cube = MeshGpu::from_vertices(gl, program, &build_cube_vertices(0.11))?;
            let octahedron = MeshGpu::from_vertices(gl, program, &build_octahedron_vertices(0.15))?;

            Ok(Self {
                program,
                cube,
                octahedron,
                u_mvp: gl.get_uniform_location(program, "u_mvp"),
                u_model: gl.get_uniform_location(program, "u_model"),
                u_inv_cam_capture: gl.get_uniform_location(program, "u_inv_cam_capture"),
                u_depth_tex: gl.get_uniform_location(program, "u_depth_tex"),
                u_occlusion_enabled: gl.get_uniform_location(program, "u_occlusion_enabled"),
                u_tan_fov_scaled: gl.get_uniform_location(program, "u_tan_fov_scaled"),
                u_d0: gl.get_uniform_location(program, "u_d0"),
                u_dx_vec: gl.get_uniform_location(program, "u_dx_vec"),
                u_dy_vec: gl.get_uniform_location(program, "u_dy_vec"),
                u_color_shift: gl.get_uniform_location(program, "u_color_shift"),
            })
        }
    }

    pub fn object_transforms(
        anchor_origin: &Mat4,
        time_sec: f32,
        pull_closer: bool,
        boost_factor: f32,
    ) -> (Mat4, Mat4) {
        let z_cube = if pull_closer { -0.78 } else { -1.25 };
        let z_octa = if pull_closer { -1.05 } else { -1.65 };
        let speed = 1.0 + boost_factor * 2.5;
        let pulse = 1.0 + boost_factor * 0.22 * (time_sec * 12.0).sin().abs();

        let cube_local = Mat4::from_translation(Vec3::new(-0.18, -0.02, z_cube))
            * Mat4::from_rotation_y(time_sec * 0.65 * speed)
            * Mat4::from_rotation_x(time_sec * 0.30 * speed)
            * Mat4::from_scale(Vec3::splat(pulse));

        let bob_y = (time_sec * 1.8).sin() * 0.045;
        let octa_local = Mat4::from_translation(Vec3::new(0.24, 0.04 + bob_y, z_octa))
            * Mat4::from_rotation_y(-time_sec * 0.85 * speed)
            * Mat4::from_scale(Vec3::splat(pulse));

        (*anchor_origin * cube_local, *anchor_origin * octa_local)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        gl: &glow::Context,
        view_proj: &Mat4,
        cube_model: &Mat4,
        octa_model: &Mat4,
        inv_cam_capture: &Mat4,
        depth_tex: &DepthTextureGpu,
        occlusion_enabled: bool,
        tan_half_h_scaled: f32,
        tan_half_v_scaled: f32,
        palette_idx: u32,
    ) {
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(true);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            gl.use_program(Some(self.program));

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
            if let Some(loc) = &self.u_color_shift {
                gl.uniform_1_f32(Some(loc), (palette_idx % 3) as f32);
            }

            self.draw_mesh(gl, &self.cube, view_proj, cube_model);
            self.draw_mesh(gl, &self.octahedron, view_proj, octa_model);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_vertex_array(None);
            gl.disable(glow::CULL_FACE);
        }
    }

    unsafe fn draw_mesh(
        &self,
        gl: &glow::Context,
        mesh: &MeshGpu,
        view_proj: &Mat4,
        model: &Mat4,
    ) {
        let mvp = *view_proj * *model;
        if let Some(loc) = &self.u_mvp {
            gl.uniform_matrix_4_f32_slice(Some(loc), false, mvp.as_ref());
        }
        if let Some(loc) = &self.u_model {
            gl.uniform_matrix_4_f32_slice(Some(loc), false, model.as_ref());
        }
        gl.bind_vertex_array(Some(mesh.vao));
        gl.draw_arrays(glow::TRIANGLES, 0, mesh.vertex_count);
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            self.cube.destroy(gl);
            self.octahedron.destroy(gl);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_meshes_have_expected_counts_and_unit_normals() {
        let cube = build_cube_vertices(0.11);
        assert_eq!(cube.len(), 36);
        assert_eq!(size_of::<Vertex3D>(), 36);
        for v in &cube {
            let n = Vec3::from_array(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }

        let octa = build_octahedron_vertices(0.15);
        assert_eq!(octa.len(), 24);
        for v in &octa {
            let n = Vec3::from_array(v.normal);
            assert!((n.length() - 1.0).abs() < 1e-4);
        }
    }
}
