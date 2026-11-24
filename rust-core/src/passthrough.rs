use glam::Mat4;
use glow::HasContext;
use std::mem::size_of;

pub const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8D65;
pub const PASSTHROUGH_DIST: f32 = 2.5;

const PASSTHROUGH_VERT: &str = r#"
precision highp float;
attribute vec3 a_pos;
attribute vec2 a_uv;
uniform mat4 u_mvp;
varying vec2 v_uv;
void main() {
    v_uv = a_uv;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const PASSTHROUGH_FRAG: &str = r#"
#extension GL_OES_EGL_image_external : require
precision mediump float;
uniform samplerExternalOES u_cam_tex;
uniform float u_has_frame;
varying vec2 v_uv;
void main() {
    if (u_has_frame < 0.5) {
        gl_FragColor = vec4(0.03, 0.04, 0.07, 1.0);
    } else {
        gl_FragColor = texture2D(u_cam_tex, v_uv);
    }
}
"#;

/// Computes the 3D virtual passthrough quad half-extents at `dist` meters so that
/// real-world objects retain a 1:1 angular size when `fov_scale == 1.0`, leaving
/// pure black AMOLED borders outside the physical camera's FOV instead of zooming in.
pub fn compute_quad_half_extents(
    tan_half_hfov: f32,
    tan_half_vfov: f32,
    dist: f32,
    fov_scale: f32,
) -> (f32, f32) {
    let clamped_scale = fov_scale.clamp(0.45, 1.25);
    let safe_h = tan_half_hfov.clamp(0.2, 2.5);
    let safe_v = tan_half_vfov.clamp(0.2, 2.5);
    (
        dist * safe_h * clamped_scale,
        dist * safe_v * clamped_scale,
    )
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PassthroughVertex {
    pos: [f32; 3],
    uv: [f32; 2],
}

pub struct PassthroughRenderer {
    program: glow::Program,
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    cam_texture: glow::Texture,
    u_mvp: Option<glow::UniformLocation>,
    u_cam_tex: Option<glow::UniformLocation>,
    u_has_frame: Option<glow::UniformLocation>,
}

impl PassthroughRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, PASSTHROUGH_VERT, PASSTHROUGH_FRAG)?;
            let vao = gl.create_vertex_array()?;
            let vbo = gl.create_buffer()?;
            let cam_texture = gl.create_texture()?;

            gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, Some(cam_texture));
            gl.tex_parameter_i32(
                GL_TEXTURE_EXTERNAL_OES,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                GL_TEXTURE_EXTERNAL_OES,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                GL_TEXTURE_EXTERNAL_OES,
                glow::TEXTURE_MIN_FILTER,
                glow::LINEAR as i32,
            );
            gl.tex_parameter_i32(
                GL_TEXTURE_EXTERNAL_OES,
                glow::TEXTURE_MAG_FILTER,
                glow::LINEAR as i32,
            );
            gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, None);

            let initial_verts = build_quad_vertices(1.0, 0.75, PASSTHROUGH_DIST, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
            gl.bind_vertex_array(Some(vao));
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
            gl.buffer_data_u8_slice(
                glow::ARRAY_BUFFER,
                as_bytes(&initial_verts),
                glow::DYNAMIC_DRAW,
            );

            let stride = size_of::<PassthroughVertex>() as i32;
            if let Some(a_pos) = gl.get_attrib_location(program, "a_pos") {
                gl.enable_vertex_attrib_array(a_pos);
                gl.vertex_attrib_pointer_f32(a_pos, 3, glow::FLOAT, false, stride, 0);
            }
            if let Some(a_uv) = gl.get_attrib_location(program, "a_uv") {
                gl.enable_vertex_attrib_array(a_uv);
                gl.vertex_attrib_pointer_f32(
                    a_uv,
                    2,
                    glow::FLOAT,
                    false,
                    stride,
                    (3 * size_of::<f32>()) as i32,
                );
            }

            gl.bind_vertex_array(None);
            gl.bind_buffer(glow::ARRAY_BUFFER, None);

            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            let u_cam_tex = gl.get_uniform_location(program, "u_cam_tex");
            let u_has_frame = gl.get_uniform_location(program, "u_has_frame");

            Ok(Self {
                program,
                vao,
                vbo,
                cam_texture,
                u_mvp,
                u_cam_tex,
                u_has_frame,
            })
        }
    }

    pub fn camera_texture_id(&self) -> u32 {
        self.cam_texture.0.get()
    }

    /// Updates the virtual camera quad vertices once per render frame before drawing left/right eyes.
    pub fn update_quad(
        &self,
        gl: &glow::Context,
        tan_half_hfov: f32,
        tan_half_vfov: f32,
        fov_scale: f32,
        uvs: [f32; 8],
    ) {
        let (hw, hh) = compute_quad_half_extents(tan_half_hfov, tan_half_vfov, PASSTHROUGH_DIST, fov_scale);
        let verts = build_quad_vertices(hw, hh, PASSTHROUGH_DIST, uvs);
        unsafe {
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo));
            gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, as_bytes(&verts));
            gl.bind_buffer(glow::ARRAY_BUFFER, None);
        }
    }

    /// Draws the zero-copy OES passthrough quad using the 120Hz predicted eye MVP.
    pub fn draw(&self, gl: &glow::Context, mvp: &Mat4, has_frame: bool) {
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.vao));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, Some(self.cam_texture));

            if let Some(loc) = &self.u_cam_tex {
                gl.uniform_1_i32(Some(loc), 0);
            }
            if let Some(loc) = &self.u_has_frame {
                gl.uniform_1_f32(Some(loc), if has_frame { 1.0 } else { 0.0 });
            }
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, mvp.as_ref());
            }

            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);

            gl.bind_texture(GL_TEXTURE_EXTERNAL_OES, None);
            gl.bind_vertex_array(None);
            gl.depth_mask(true);
            gl.enable(glow::DEPTH_TEST);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_texture(self.cam_texture);
            gl.delete_buffer(self.vbo);
            gl.delete_vertex_array(self.vao);
            gl.delete_program(self.program);
        }
    }
}

fn build_quad_vertices(
    half_w: f32,
    half_h: f32,
    dist: f32,
    uvs: [f32; 8],
) -> [PassthroughVertex; 4] {
    [
        PassthroughVertex {
            pos: [-half_w, -half_h, -dist],
            uv: [uvs[0], uvs[1]],
        },
        PassthroughVertex {
            pos: [half_w, -half_h, -dist],
            uv: [uvs[2], uvs[3]],
        },
        PassthroughVertex {
            pos: [-half_w, half_h, -dist],
            uv: [uvs[4], uvs[5]],
        },
        PassthroughVertex {
            pos: [half_w, half_h, -dist],
            uv: [uvs[6], uvs[7]],
        },
    ]
}

unsafe fn as_bytes<T>(slice: &[T]) -> &[u8] {
    std::slice::from_raw_parts(slice.as_ptr() as *const u8, std::mem::size_of_val(slice))
}

pub unsafe fn compile_program(
    gl: &glow::Context,
    vert_src: &str,
    frag_src: &str,
) -> Result<glow::Program, String> {
    let vs = gl.create_shader(glow::VERTEX_SHADER)?;
    gl.shader_source(vs, vert_src);
    gl.compile_shader(vs);
    if !gl.get_shader_compile_status(vs) {
        let log = gl.get_shader_info_log(vs);
        gl.delete_shader(vs);
        return Err(format!("Vertex shader compile error: {}", log));
    }

    let fs = gl.create_shader(glow::FRAGMENT_SHADER)?;
    gl.shader_source(fs, frag_src);
    gl.compile_shader(fs);
    if !gl.get_shader_compile_status(fs) {
        let log = gl.get_shader_info_log(fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);
        return Err(format!("Fragment shader compile error: {}", log));
    }

    let prog = gl.create_program()?;
    gl.attach_shader(prog, vs);
    gl.attach_shader(prog, fs);
    gl.link_program(prog);
    gl.delete_shader(vs);
    gl.delete_shader(fs);

    if !gl.get_program_link_status(prog) {
        let log = gl.get_program_info_log(prog);
        gl.delete_program(prog);
        return Err(format!("Shader program link error: {}", log));
    }

    Ok(prog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec4;

    #[test]
    fn quad_preserves_1_to_1_angular_scale_inside_wider_vr_fov() {
        // Camera ~68 deg HFOV, ~55 deg VFOV; VR box lens ~85 deg VFOV.
        let tan_cam_h = 34.0_f32.to_radians().tan();
        let tan_cam_v = 27.5_f32.to_radians().tan();
        let (hw, hh) = compute_quad_half_extents(tan_cam_h, tan_cam_v, PASSTHROUGH_DIST, 1.0);

        let vr_proj = Mat4::perspective_rh_gl(85.0_f32.to_radians(), 1200.0 / 1080.0, 0.05, 50.0);
        let clip = vr_proj * Vec4::new(hw, hh, -PASSTHROUGH_DIST, 1.0);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;

        // Narrower camera window stays inside [-1, 1] NDC leaving pure black AMOLED surround.
        assert!(ndc_x > 0.4 && ndc_x < 0.95);
        assert!(ndc_y > 0.4 && ndc_y < 0.95);
        // Vertical NDC ratio should exactly match tan(cam_vfov/2) / tan(vr_vfov/2).
        let expected_ndc_y = tan_cam_v / 42.5_f32.to_radians().tan();
        assert!((ndc_y - expected_ndc_y).abs() < 1e-4);
    }
}
