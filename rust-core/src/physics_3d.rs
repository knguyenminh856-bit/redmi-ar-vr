use crate::passthrough::compile_program;
use crate::room_scanner::{AabbCollider, PlaneMarker, SurfaceKind};
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::mem::size_of;

pub const MAX_BODIES: usize = 16;
const GRAVITY_Y: f32 = 9.81;

const PHYS_VERT: &str = r#"
precision highp float;
attribute vec3 a_pos;
attribute vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
varying vec3 v_normal_w;
void main() {
    v_normal_w = normalize(mat3(u_model) * a_normal);
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
"#;

const PHYS_FRAG: &str = r#"
precision mediump float;
uniform vec4 u_color;
uniform float u_flash;
uniform float u_is_shadow;
varying vec3 v_normal_w;
void main() {
    if (u_is_shadow > 0.5) {
        gl_FragColor = u_color;
        return;
    }
    vec3 n = normalize(v_normal_w);
    vec3 light_dir = normalize(vec3(0.45, 0.85, 0.55));
    float diff = max(dot(n, light_dir), 0.0);
    float hemi = 0.36 + 0.22 * n.y;
    vec3 view_dir = normalize(vec3(0.0, 0.20, 1.0));
    vec3 half_vec = normalize(light_dir + view_dir);
    float spec = pow(max(dot(n, half_vec), 0.0), 28.0) * 0.32;
    vec3 lit = u_color.rgb * (hemi + diff * 0.72) + vec3(spec) + vec3(0.35, 0.45, 0.55) * u_flash;
    gl_FragColor = vec4(lit, u_color.a);
}
"#;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BodyShape {
    Cube,
    Sphere,
}

#[derive(Clone, Debug)]
pub struct PhysicsBody {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Quat,
    pub ang_vel: Vec3,
    pub radius: f32,
    pub shape: BodyShape,
    pub color: [f32; 3],
    pub restitution: f32,
    pub impact_flash: f32,
    pub grounded: bool,
    pub shadow_y: f32,
}

const BODY_PALETTE: [[f32; 3]; 6] = [
    [0.18, 0.92, 0.86], // Cyan
    [1.00, 0.62, 0.22], // Tangerine
    [0.42, 0.96, 0.56], // Mint
    [0.82, 0.48, 1.00], // Violet
    [1.00, 0.38, 0.58], // Coral pink
    [0.98, 0.84, 0.26], // Gold
];

pub struct PhysicsWorld {
    pub bodies: Vec<PhysicsBody>,
    next_idx: usize,
    prev_hand_pos: Option<Vec3>,
}

impl Default for PhysicsWorld {
    fn default() -> Self {
        Self {
            bodies: Vec::with_capacity(MAX_BODIES),
            next_idx: 0,
            prev_hand_pos: None,
        }
    }
}

impl PhysicsWorld {
    pub fn clear(&mut self) {
        self.bodies.clear();
    }

    pub fn spawn_throw(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        speed: f32,
        shape_override: Option<BodyShape>,
    ) {
        let idx = self.next_idx;
        self.next_idx = self.next_idx.wrapping_add(1);

        let shape = shape_override.unwrap_or(if idx % 2 == 0 {
            BodyShape::Cube
        } else {
            BodyShape::Sphere
        });
        let (radius, restitution) = match shape {
            BodyShape::Cube => (0.052, 0.34),
            BodyShape::Sphere => (0.056, 0.58),
        };
        let color = BODY_PALETTE[idx % BODY_PALETTE.len()];
        let throw_dir = dir.normalize_or_zero();
        let vel = throw_dir * speed + Vec3::new(0.0, 0.38, 0.0);
        let spin_sign = if idx % 2 == 0 { 1.0 } else { -1.0 };
        let ang_vel = Vec3::new(2.2 * spin_sign, 1.5, -1.8 * spin_sign);

        if self.bodies.len() >= MAX_BODIES {
            self.bodies.remove(0);
        }
        self.bodies.push(PhysicsBody {
            pos: origin,
            vel,
            rot: Quat::IDENTITY,
            ang_vel,
            radius,
            shape,
            color,
            restitution,
            impact_flash: 0.45,
            grounded: false,
            shadow_y: origin.y - 1.2,
        });
    }

    pub fn step(
        &mut self,
        dt_sec: f32,
        planes: &[PlaneMarker],
        aabbs: &[AabbCollider],
        fallback_floor_y: f32,
        hand_world_pos: Option<Vec3>,
    ) {
        let dt = dt_sec.clamp(0.001, 0.025);
        let hand_vel = match (self.prev_hand_pos, hand_world_pos) {
            (Some(prev), Some(curr)) => ((curr - prev) / dt).clamp_length_max(4.0),
            _ => Vec3::ZERO,
        };
        self.prev_hand_pos = hand_world_pos;

        if self.bodies.is_empty() {
            return;
        }

        let sub_dt = dt * 0.5;
        for _sub in 0..2 {
            for b in &mut self.bodies {
                b.grounded = false;
                b.vel.y -= GRAVITY_Y * sub_dt;
                b.vel *= (1.0 - 0.14 * sub_dt).clamp(0.90, 1.0);
                b.pos += b.vel * sub_dt;

                let ang_speed = b.ang_vel.length();
                if ang_speed > 1e-4 {
                    let dq = Quat::from_axis_angle(b.ang_vel / ang_speed, ang_speed * sub_dt);
                    b.rot = (dq * b.rot).normalize();
                }
                b.impact_flash = (b.impact_flash - sub_dt * 2.8).max(0.0);

                // 1. ARCore plane collisions (Floor, Table, Seat, Ceiling, Wall, VerticalObject)
                for plane in planes {
                    let n = (plane.rot * Vec3::Y).normalize_or_zero();
                    if n.length_squared() < 1e-4 {
                        continue;
                    }
                    let ax_x = plane.rot * Vec3::X;
                    let ax_z = plane.rot * Vec3::Z;
                    let rel = b.pos - plane.pos;
                    let extra = if plane.kind == SurfaceKind::Floor {
                        0.85
                    } else {
                        0.14
                    };
                    if rel.dot(ax_x).abs() > plane.extent_x * 0.5 + extra
                        || rel.dot(ax_z).abs() > plane.extent_z * 0.5 + extra
                    {
                        continue;
                    }

                    let signed_dist = rel.dot(n);
                    let is_wall =
                        matches!(plane.kind, SurfaceKind::Wall | SurfaceKind::VerticalObject);
                    if is_wall {
                        if signed_dist.abs() < b.radius {
                            let push_n = if signed_dist >= 0.0 { n } else { -n };
                            let pen = b.radius - signed_dist.abs();
                            b.pos += push_n * pen;
                            apply_contact_response(b, push_n, sub_dt);
                        }
                    } else if signed_dist < b.radius && signed_dist > -0.22 {
                        let pen = b.radius - signed_dist;
                        b.pos += n * pen;
                        apply_contact_response(b, n, sub_dt);
                        if n.y > 0.5 {
                            b.grounded = true;
                        }
                    }
                }

                // 2. 2.5D Voxel & Extruded Object AABB Colliders
                for box_col in aabbs {
                    let closest = b.pos.clamp(box_col.min, box_col.max);
                    let diff = b.pos - closest;
                    let d2 = diff.length_squared();
                    if d2 < b.radius * b.radius {
                        let (push_n, pen) = if d2 > 1e-8 {
                            let d = d2.sqrt();
                            (diff / d, b.radius - d)
                        } else {
                            // Center inside box: resolve along shallowest exit face
                            let faces = [
                                (-Vec3::X, b.pos.x - box_col.min.x),
                                (Vec3::X, box_col.max.x - b.pos.x),
                                (-Vec3::Y, b.pos.y - box_col.min.y),
                                (Vec3::Y, box_col.max.y - b.pos.y),
                                (-Vec3::Z, b.pos.z - box_col.min.z),
                                (Vec3::Z, box_col.max.z - b.pos.z),
                            ];
                            let (best_n, best_d) = faces
                                .into_iter()
                                .min_by(|a, c| a.1.total_cmp(&c.1))
                                .unwrap_or((Vec3::Y, 0.0));
                            (best_n, best_d + b.radius)
                        };
                        b.pos += push_n * pen;
                        apply_contact_response(b, push_n, sub_dt);
                        if push_n.y > 0.5 {
                            b.grounded = true;
                        }
                    }
                }

                // 3. Fallback floor plane prevents dropping to infinity before floor is mapped
                if b.pos.y - b.radius < fallback_floor_y {
                    b.pos.y = fallback_floor_y + b.radius;
                    apply_contact_response(b, Vec3::Y, sub_dt);
                    b.grounded = true;
                }

                // 4. Physical hand / fingertip slapping impulse
                if let Some(hand_pos) = hand_world_pos {
                    let reach = b.radius + 0.085;
                    let diff = b.pos - hand_pos;
                    let d2 = diff.length_squared();
                    if d2 < reach * reach {
                        let d = d2.sqrt().max(1e-4);
                        let push_n = diff / d;
                        b.pos += push_n * (reach - d);
                        let push_vel = hand_vel * 0.85 + push_n * 0.75;
                        b.vel += push_vel * 0.5;
                        b.ang_vel += push_n.cross(hand_vel) * 4.0;
                        b.impact_flash = b.impact_flash.max(0.85);
                    }
                }
            }

            // 5. Pairwise body-vs-body collisions
            let len = self.bodies.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    let delta = self.bodies[j].pos - self.bodies[i].pos;
                    let min_dist = self.bodies[i].radius + self.bodies[j].radius;
                    let d2 = delta.length_squared();
                    if d2 < min_dist * min_dist {
                        let d = d2.sqrt().max(1e-4);
                        let n = delta / d;
                        let pen = (min_dist - d) * 0.5;
                        self.bodies[i].pos -= n * pen;
                        self.bodies[j].pos += n * pen;

                        let rel_vn = (self.bodies[j].vel - self.bodies[i].vel).dot(n);
                        if rel_vn < 0.0 {
                            let e = self.bodies[i]
                                .restitution
                                .min(self.bodies[j].restitution)
                                * 0.85;
                            let impulse = -(1.0 + e) * rel_vn * 0.5;
                            self.bodies[i].vel -= n * impulse;
                            self.bodies[j].vel += n * impulse;
                            if -rel_vn > 0.5 {
                                self.bodies[i].impact_flash = self.bodies[i].impact_flash.max(0.75);
                                self.bodies[j].impact_flash = self.bodies[j].impact_flash.max(0.75);
                            }
                        }
                    }
                }
            }
        }

        // Update contact shadow height for every body once per frame
        for b in &mut self.bodies {
            let mut best_y = fallback_floor_y;
            for plane in planes {
                if !matches!(
                    plane.kind,
                    SurfaceKind::Floor | SurfaceKind::Table | SurfaceKind::Seat
                ) || plane.pos.y > b.pos.y + 0.02
                {
                    continue;
                }
                let rel = b.pos - plane.pos;
                let extra = if plane.kind == SurfaceKind::Floor {
                    0.85
                } else {
                    0.12
                };
                if rel.dot(plane.rot * Vec3::X).abs() <= plane.extent_x * 0.5 + extra
                    && rel.dot(plane.rot * Vec3::Z).abs() <= plane.extent_z * 0.5 + extra
                {
                    best_y = best_y.max(plane.pos.y);
                }
            }
            for box_col in aabbs {
                if box_col.max.y <= b.pos.y + 0.02
                    && b.pos.x >= box_col.min.x - 0.04
                    && b.pos.x <= box_col.max.x + 0.04
                    && b.pos.z >= box_col.min.z - 0.04
                    && b.pos.z <= box_col.max.z + 0.04
                {
                    best_y = best_y.max(box_col.max.y);
                }
            }
            b.shadow_y = best_y.min(b.pos.y - b.radius + 0.01);
        }
    }
}

fn apply_contact_response(b: &mut PhysicsBody, n: Vec3, sub_dt: f32) {
    let vn = b.vel.dot(n);
    if vn < 0.0 {
        let bounce = if -vn < 0.35 { 0.0 } else { b.restitution };
        b.vel -= (1.0 + bounce) * vn * n;
        if -vn > 0.45 {
            b.impact_flash = b.impact_flash.max((-vn / 2.8).clamp(0.25, 1.0));
        }

        let vt = b.vel - b.vel.dot(n) * n;
        let friction = (1.0 - 4.2 * sub_dt).clamp(0.55, 0.98);
        b.vel = b.vel.dot(n) * n + vt * friction;

        // Visual rolling spin aligned with surface tangent velocity
        if vt.length_squared() > 1e-4 {
            let roll = n.cross(vt) / b.radius.max(0.02);
            b.ang_vel = b.ang_vel * 0.72 + roll * 0.28;
        } else {
            b.ang_vel *= 0.85;
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PhysVertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

struct PhysMeshGpu {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    vert_count: i32,
}

impl PhysMeshGpu {
    unsafe fn from_vertices(
        gl: &glow::Context,
        program: glow::Program,
        verts: &[PhysVertex],
    ) -> Result<Self, String> {
        let vao = gl.create_vertex_array()?;
        let vbo = gl.create_buffer()?;
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        let bytes = std::slice::from_raw_parts(
            verts.as_ptr() as *const u8,
            verts.len() * size_of::<PhysVertex>(),
        );
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytes, glow::STATIC_DRAW);

        let stride = size_of::<PhysVertex>() as i32;
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

        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok(Self {
            vao,
            vbo,
            vert_count: verts.len() as i32,
        })
    }

    unsafe fn destroy(&self, gl: &glow::Context) {
        gl.delete_buffer(self.vbo);
        gl.delete_vertex_array(self.vao);
    }
}

fn build_unit_cube_verts() -> Vec<PhysVertex> {
    let faces: [([Vec3; 4], Vec3); 6] = [
        ([Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0)], Vec3::Z),
        ([Vec3::new(1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, -1.0)], -Vec3::Z),
        ([Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, 1.0, -1.0), Vec3::new(1.0, 1.0, 1.0)], Vec3::X),
        ([Vec3::new(-1.0, -1.0, -1.0), Vec3::new(-1.0, -1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, -1.0)], -Vec3::X),
        ([Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, -1.0), Vec3::new(-1.0, 1.0, -1.0)], Vec3::Y),
        ([Vec3::new(-1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, -1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0)], -Vec3::Y),
    ];
    let mut out = Vec::with_capacity(36);
    for (q, n) in faces {
        let narr = n.to_array();
        for &idx in &[0usize, 1, 2, 0, 2, 3] {
            out.push(PhysVertex {
                pos: q[idx].to_array(),
                normal: narr,
            });
        }
    }
    out
}

fn build_unit_sphere_verts() -> Vec<PhysVertex> {
    let mut tris = vec![
        [Vec3::Y, Vec3::Z, Vec3::X],
        [Vec3::Y, Vec3::X, -Vec3::Z],
        [Vec3::Y, -Vec3::Z, -Vec3::X],
        [Vec3::Y, -Vec3::X, Vec3::Z],
        [-Vec3::Y, Vec3::X, Vec3::Z],
        [-Vec3::Y, -Vec3::Z, Vec3::X],
        [-Vec3::Y, -Vec3::X, -Vec3::Z],
        [-Vec3::Y, Vec3::Z, -Vec3::X],
    ];
    for _ in 0..2 {
        let mut next = Vec::with_capacity(tris.len() * 4);
        for [a, b, c] in tris {
            let ab = (a + b).normalize();
            let bc = (b + c).normalize();
            let ca = (c + a).normalize();
            next.push([a, ab, ca]);
            next.push([ab, b, bc]);
            next.push([ca, bc, c]);
            next.push([ab, bc, ca]);
        }
        tris = next;
    }

    let mut out = Vec::with_capacity(tris.len() * 3);
    for [a, b, c] in tris {
        let mid = (a + b + c).normalize_or_zero();
        let (v0, v1, v2) = if (b - a).cross(c - a).dot(mid) < 0.0 {
            (a, c, b)
        } else {
            (a, b, c)
        };
        for p in [v0, v1, v2] {
            out.push(PhysVertex {
                pos: p.to_array(),
                normal: p.to_array(),
            });
        }
    }
    out
}

fn build_shadow_disc_verts() -> Vec<PhysVertex> {
    let segs = 12usize;
    let mut out = Vec::with_capacity(segs * 3);
    for i in 0..segs {
        let a0 = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let a1 = ((i + 1) as f32 / segs as f32) * std::f32::consts::TAU;
        out.push(PhysVertex {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        });
        out.push(PhysVertex {
            pos: [a0.cos(), 0.0, a0.sin()],
            normal: [0.0, 1.0, 0.0],
        });
        out.push(PhysVertex {
            pos: [a1.cos(), 0.0, a1.sin()],
            normal: [0.0, 1.0, 0.0],
        });
    }
    out
}

pub struct PhysicsRenderer {
    program: glow::Program,
    cube: PhysMeshGpu,
    sphere: PhysMeshGpu,
    shadow: PhysMeshGpu,
    u_mvp: Option<glow::UniformLocation>,
    u_model: Option<glow::UniformLocation>,
    u_color: Option<glow::UniformLocation>,
    u_flash: Option<glow::UniformLocation>,
    u_is_shadow: Option<glow::UniformLocation>,
}

impl PhysicsRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, PHYS_VERT, PHYS_FRAG)?;
            let cube = PhysMeshGpu::from_vertices(gl, program, &build_unit_cube_verts())?;
            let sphere = PhysMeshGpu::from_vertices(gl, program, &build_unit_sphere_verts())?;
            let shadow = PhysMeshGpu::from_vertices(gl, program, &build_shadow_disc_verts())?;
            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            let u_model = gl.get_uniform_location(program, "u_model");
            let u_color = gl.get_uniform_location(program, "u_color");
            let u_flash = gl.get_uniform_location(program, "u_flash");
            let u_is_shadow = gl.get_uniform_location(program, "u_is_shadow");
            Ok(Self {
                program,
                cube,
                sphere,
                shadow,
                u_mvp,
                u_model,
                u_color,
                u_flash,
                u_is_shadow,
            })
        }
    }

    pub fn draw(&self, gl: &glow::Context, view_proj: &Mat4, bodies: &[PhysicsBody]) {
        if bodies.is_empty() {
            return;
        }
        unsafe {
            gl.use_program(Some(self.program));

            // Pass 1: Soft contact shadows projected onto surfaces below each body
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            if let Some(loc) = &self.u_is_shadow {
                gl.uniform_1_f32(Some(loc), 1.0);
            }
            if let Some(loc) = &self.u_flash {
                gl.uniform_1_f32(Some(loc), 0.0);
            }
            gl.bind_vertex_array(Some(self.shadow.vao));
            for b in bodies {
                let h = (b.pos.y - b.radius - b.shadow_y).max(0.0);
                let fade = (1.0 - h / 1.4).clamp(0.0, 1.0);
                if fade <= 0.02 {
                    continue;
                }
                let s_rad = b.radius * (1.15 - fade * 0.25).clamp(0.55, 1.2);
                let model = Mat4::from_translation(Vec3::new(b.pos.x, b.shadow_y + 0.006, b.pos.z))
                    * Mat4::from_scale(Vec3::new(s_rad, 1.0, s_rad));
                let mvp = *view_proj * model;
                if let Some(loc) = &self.u_mvp {
                    gl.uniform_matrix_4_f32_slice(Some(loc), false, mvp.as_ref());
                }
                if let Some(loc) = &self.u_model {
                    gl.uniform_matrix_4_f32_slice(Some(loc), false, model.as_ref());
                }
                if let Some(loc) = &self.u_color {
                    gl.uniform_4_f32(Some(loc), 0.0, 0.0, 0.0, 0.36 * fade);
                }
                gl.draw_arrays(glow::TRIANGLES, 0, self.shadow.vert_count);
            }
            gl.disable(glow::BLEND);

            // Pass 2: Shaded rigid bodies
            gl.depth_mask(true);
            gl.enable(glow::CULL_FACE);
            gl.cull_face(glow::BACK);
            if let Some(loc) = &self.u_is_shadow {
                gl.uniform_1_f32(Some(loc), 0.0);
            }
            for b in bodies {
                let mesh = match b.shape {
                    BodyShape::Cube => &self.cube,
                    BodyShape::Sphere => &self.sphere,
                };
                let scale = match b.shape {
                    BodyShape::Cube => b.radius * 0.90,
                    BodyShape::Sphere => b.radius,
                };
                let model = Mat4::from_rotation_translation(b.rot, b.pos)
                    * Mat4::from_scale(Vec3::splat(scale));
                let mvp = *view_proj * model;
                if let Some(loc) = &self.u_mvp {
                    gl.uniform_matrix_4_f32_slice(Some(loc), false, mvp.as_ref());
                }
                if let Some(loc) = &self.u_model {
                    gl.uniform_matrix_4_f32_slice(Some(loc), false, model.as_ref());
                }
                if let Some(loc) = &self.u_color {
                    gl.uniform_4_f32(Some(loc), b.color[0], b.color[1], b.color[2], 1.0);
                }
                if let Some(loc) = &self.u_flash {
                    gl.uniform_1_f32(Some(loc), b.impact_flash);
                }
                gl.bind_vertex_array(Some(mesh.vao));
                gl.draw_arrays(glow::TRIANGLES, 0, mesh.vert_count);
            }

            gl.bind_vertex_array(None);
            gl.disable(glow::CULL_FACE);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            self.cube.destroy(gl);
            self.sphere.destroy(gl);
            self.shadow.destroy(gl);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gravity_drops_body_onto_floor_and_stops_cleanly() {
        let mut world = PhysicsWorld::default();
        world.spawn_throw(Vec3::new(0.0, 0.2, -1.2), Vec3::ZERO, 0.0, Some(BodyShape::Cube));

        let floor = vec![PlaneMarker {
            kind: SurfaceKind::Floor,
            pos: Vec3::new(0.0, -0.8, -1.2),
            rot: Quat::IDENTITY,
            extent_x: 3.0,
            extent_z: 3.0,
        }];

        for _ in 0..180 {
            world.step(1.0 / 60.0, &floor, &[], -1.5, None);
        }
        let b = &world.bodies[0];
        assert!(b.grounded);
        assert!((b.pos.y - (-0.8 + b.radius)).abs() < 0.015);
        assert!(b.vel.length() < 0.08);
    }

    #[test]
    fn wall_blocks_and_reverses_thrown_body() {
        let mut world = PhysicsWorld::default();
        world.spawn_throw(Vec3::new(0.0, 0.0, -0.4), Vec3::new(0.0, 0.0, -1.0), 3.0, Some(BodyShape::Sphere));

        let wall_rot = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
        let wall = vec![PlaneMarker {
            kind: SurfaceKind::Wall,
            pos: Vec3::new(0.0, 0.0, -0.9),
            rot: wall_rot,
            extent_x: 2.5,
            extent_z: 2.0,
        }];

        let mut hit_wall = false;
        for _ in 0..35 {
            world.step(1.0 / 60.0, &wall, &[], -1.5, None);
            if world.bodies[0].vel.z > 0.1 {
                hit_wall = true;
            }
            // Body must stay on our side of wall (-0.9) and never tunnel past it
            assert!(world.bodies[0].pos.z >= -0.9);
        }
        assert!(hit_wall);
    }

    #[test]
    fn voxel_and_object_25d_box_stops_falling_and_side_motion() {
        let mut world = PhysicsWorld::default();
        world.spawn_throw(Vec3::new(0.0, 0.35, -1.0), Vec3::ZERO, 0.0, Some(BodyShape::Sphere));

        let box_col = vec![AabbCollider {
            min: Vec3::new(-0.25, -0.5, -1.25),
            max: Vec3::new(0.25, 0.0, -0.75),
            is_object_box: true,
        }];

        for _ in 0..120 {
            world.step(1.0 / 60.0, &[], &box_col, -1.5, None);
        }
        let b = &world.bodies[0];
        assert!(b.grounded);
        // Should rest right on top of the 2.5D object box (y = 0.0 + radius), NOT drop to -1.5m
        assert!((b.pos.y - b.radius).abs() < 0.015);
    }
}
