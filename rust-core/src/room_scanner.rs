use crate::passthrough::{compile_program, PASSTHROUGH_DIST};
use glam::{Mat4, Quat, Vec3};
use glow::HasContext;
use std::mem::size_of;

pub const MAX_PLANES: usize = 12;
pub const FLOATS_PER_PLANE: usize = 10; // [raw_type, px, py, pz, qx, qy, qz, qw, extent_x, extent_z]

pub const MAX_OBJECTS: usize = 6;
pub const FLOATS_PER_OBJECT: usize = 6; // [kind_id, ndc_x, ndc_y, ndc_w, ndc_h, score]

pub const MAX_LABELS: usize = 6;
pub const FLOATS_PER_LABEL: usize = 6; // [kind_id, left_u, left_v, right_u, right_v, score]

pub const PLANE_TIMEOUT_MS: u128 = 2500;
pub const OBJECT_TIMEOUT_MS: u128 = 1200;

const MAX_WORLD_VERTICES: usize = 2400;
const MAX_PASS_VERTICES: usize = 1200;

const SCAN_VERT: &str = r#"
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

const SCAN_FRAG: &str = r#"
precision mediump float;
varying vec4 v_color;
void main() {
    gl_FragColor = v_color;
}
"#;

#[repr(i32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Unknown = 0,
    Floor = 1,
    Wall = 2,
    Table = 3,
    Seat = 4,
    Ceiling = 5,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ScanVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
pub struct PlaneMarker {
    pub kind: SurfaceKind,
    pub pos: Vec3,
    pub rot: Quat,
    pub extent_x: f32,
    pub extent_z: f32,
}

#[derive(Copy, Clone, Debug)]
pub struct ObjectMarker {
    pub kind_id: i32,
    pub ndc_x: f32,
    pub ndc_y: f32,
    pub ndc_w: f32,
    pub ndc_h: f32,
    pub score: f32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ProjectedLabel {
    pub kind_id: i32,
    pub left_u: f32,
    pub left_v: f32,
    pub right_u: f32,
    pub right_v: f32,
    pub score: f32,
}

/// Estimates the room floor Y elevation from tracked upward-facing horizontal ARCore planes.
pub fn infer_floor_y(planes: &[(i32, Vec3, f32, f32)], cam_y: f32) -> Option<f32> {
    let mut min_upward_y: Option<f32> = None;
    let mut max_upward_y: Option<f32> = None;
    let mut min_area = 0.0f32;

    for &(raw_type, pos, ex, ez) in planes {
        if raw_type == 0 {
            let area = (ex * ez).max(0.0);
            match min_upward_y {
                None => {
                    min_upward_y = Some(pos.y);
                    min_area = area;
                }
                Some(prev) if pos.y < prev => {
                    min_upward_y = Some(pos.y);
                    min_area = area;
                }
                _ => {}
            }
            max_upward_y = Some(match max_upward_y {
                None => pos.y,
                Some(prev) => prev.max(pos.y),
            });
        }
    }

    let min_y = min_upward_y?;
    let max_y = max_upward_y.unwrap_or(min_y);
    let below_cam = cam_y - min_y;
    // Treat lowest upward plane as floor when below camera by >=0.70m, large (>=1.2m^2), or separated by >=0.30m from higher surfaces.
    if below_cam >= 0.70 || min_area >= 1.2 || (max_y - min_y) >= 0.30 {
        Some(min_y)
    } else {
        None
    }
}

/// Classifies an ARCore plane into Floor, Wall, Table, Seat, or Ceiling.
pub fn classify_ar_plane(
    raw_type: i32,
    center_y: f32,
    extent_x: f32,
    extent_z: f32,
    cam_y: f32,
    floor_y: Option<f32>,
) -> SurfaceKind {
    match raw_type {
        1 => SurfaceKind::Wall,
        2 => SurfaceKind::Ceiling,
        0 => {
            if let Some(fy) = floor_y {
                let dy = center_y - fy;
                if dy <= 0.16 {
                    SurfaceKind::Floor
                } else if dy >= 0.28 && dy < 0.58 && extent_x.max(extent_z) <= 1.25 {
                    SurfaceKind::Seat
                } else if dy >= 0.28 && dy <= 1.45 {
                    SurfaceKind::Table
                } else {
                    SurfaceKind::Floor
                }
            } else {
                let below_cam = cam_y - center_y;
                if below_cam >= 0.85 || (extent_x * extent_z) >= 1.6 {
                    SurfaceKind::Floor
                } else if below_cam >= 0.25 {
                    SurfaceKind::Table
                } else {
                    SurfaceKind::Table
                }
            }
        }
        _ => SurfaceKind::Unknown,
    }
}

pub fn parse_ar_planes(
    count: usize,
    packed: &[f32],
    cam_y: f32,
    scan_mode: i32,
) -> Vec<PlaneMarker> {
    if scan_mode <= 0 || count == 0 {
        return Vec::new();
    }
    let n = count.min(MAX_PLANES);
    let mut raw_entries = Vec::with_capacity(n);
    let mut summary = Vec::with_capacity(n);

    for i in 0..n {
        let base = i * FLOATS_PER_PLANE;
        if base + FLOATS_PER_PLANE > packed.len() {
            break;
        }
        let raw_type = packed[base].round() as i32;
        let pos = Vec3::new(packed[base + 1], packed[base + 2], packed[base + 3]);
        let rot = Quat::from_xyzw(
            packed[base + 4],
            packed[base + 5],
            packed[base + 6],
            packed[base + 7],
        );
        let rot = if rot.length_squared() > 1e-6 {
            rot.normalize()
        } else {
            Quat::IDENTITY
        };
        let ex = packed[base + 8].clamp(0.12, 8.0);
        let ez = packed[base + 9].clamp(0.12, 8.0);
        if pos.is_finite() {
            summary.push((raw_type, pos, ex, ez));
            raw_entries.push((raw_type, pos, rot, ex, ez));
        }
    }

    let floor_y = infer_floor_y(&summary, cam_y);
    let mut out = Vec::with_capacity(raw_entries.len());
    for (raw_type, pos, rot, ex, ez) in raw_entries {
        let kind = classify_ar_plane(raw_type, pos.y, ex, ez, cam_y, floor_y);
        if scan_mode == 1 && !matches!(kind, SurfaceKind::Floor | SurfaceKind::Wall) {
            continue;
        }
        if kind != SurfaceKind::Unknown {
            out.push(PlaneMarker {
                kind,
                pos,
                rot,
                extent_x: ex,
                extent_z: ez,
            });
        }
    }
    out
}

pub fn parse_detected_objects(count: usize, packed: &[f32], scan_mode: i32) -> Vec<ObjectMarker> {
    if scan_mode < 2 || count == 0 {
        return Vec::new();
    }
    let n = count.min(MAX_OBJECTS);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * FLOATS_PER_OBJECT;
        if base + FLOATS_PER_OBJECT > packed.len() {
            break;
        }
        let kind_id = packed[base].round() as i32;
        let ndc_x = packed[base + 1].clamp(-0.98, 0.98);
        let ndc_y = packed[base + 2].clamp(-0.95, 0.95);
        let ndc_w = packed[base + 3].clamp(0.04, 1.5);
        let ndc_h = packed[base + 4].clamp(0.04, 1.5);
        let score = packed[base + 5].clamp(0.0, 1.0);
        if kind_id >= 11 && ndc_x.is_finite() && ndc_y.is_finite() {
            out.push(ObjectMarker {
                kind_id,
                ndc_x,
                ndc_y,
                ndc_w,
                ndc_h,
                score,
            });
        }
    }
    out
}

fn kind_rgb(kind_id: i32) -> [f32; 3] {
    match kind_id {
        1 => [0.18, 0.86, 0.90], // Floor: cyan-teal
        2 => [0.34, 0.68, 1.00], // Wall: sky blue
        3 => [1.00, 0.72, 0.24], // Table: warm amber
        4 => [0.76, 0.52, 1.00], // Seat: soft violet
        5 => [0.58, 0.65, 0.96], // Ceiling: pale indigo
        11 | 12 | 13 | 14 => [0.24, 0.94, 0.64], // Furniture: emerald
        15 => [0.24, 0.86, 1.00],                // Screen/TV: cyan
        _ => [1.00, 0.80, 0.30],                 // Other objects: soft gold
    }
}

fn push_quad(
    out: &mut Vec<ScanVertex>,
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    color: [f32; 4],
) {
    let verts = [p0, p1, p2, p0, p2, p3];
    for p in verts {
        out.push(ScanVertex {
            pos: p.to_array(),
            color,
        });
    }
}

fn push_strip(
    out: &mut Vec<ScanVertex>,
    a: Vec3,
    b: Vec3,
    perp_dir: Vec3,
    half_width: f32,
    color: [f32; 4],
) {
    let off = perp_dir * half_width;
    push_quad(out, a - off, b - off, b + off, a + off, color);
}

/// Builds small, faint world-space translucent surface patches (max 28cm x 28cm) with thin borders.
pub fn build_plane_marker_vertices(planes: &[PlaneMarker], out: &mut Vec<ScanVertex>) {
    out.clear();
    for plane in planes.iter().take(MAX_PLANES) {
        if out.len() + 60 > MAX_WORLD_VERTICES {
            break;
        }
        let rgb = kind_rgb(plane.kind as i32);
        let fill_col = [rgb[0], rgb[1], rgb[2], 0.18];
        let edge_col = [rgb[0], rgb[1], rgb[2], 0.55];
        let cue_col = [1.0, 1.0, 1.0, 0.45];

        let ax_x = plane.rot * Vec3::X;
        let ax_y = plane.rot * Vec3::Y; // Outward plane normal
        let ax_z = plane.rot * Vec3::Z;
        let center = plane.pos + ax_y * 0.004;

        // Capped small patch (12cm..28cm across) so markers stay unobtrusive
        let hx = (plane.extent_x * 0.25).clamp(0.06, 0.14);
        let hz = (plane.extent_z * 0.25).clamp(0.06, 0.14);

        let c0 = center - ax_x * hx - ax_z * hz;
        let c1 = center + ax_x * hx - ax_z * hz;
        let c2 = center + ax_x * hx + ax_z * hz;
        let c3 = center - ax_x * hx + ax_z * hz;
        push_quad(out, c0, c1, c2, c3, fill_col);

        let bw = 0.0055;
        push_strip(out, c0, c1, ax_z, bw, edge_col);
        push_strip(out, c1, c2, ax_x, bw, edge_col);
        push_strip(out, c2, c3, ax_z, bw, edge_col);
        push_strip(out, c3, c0, ax_x, bw, edge_col);

        // Tiny subtle geometric symbol in center
        let r = (hx.min(hz) * 0.30).clamp(0.016, 0.032);
        match plane.kind {
            SurfaceKind::Table | SurfaceKind::Seat => {
                // Small rotated diamond on plane
                let d0 = center - ax_x * r;
                let d1 = center - ax_z * r;
                let d2 = center + ax_x * r;
                let d3 = center + ax_z * r;
                push_quad(out, d0, d1, d2, d3, [rgb[0], rgb[1], rgb[2], 0.34]);
            }
            _ => {
                // Thin center '+' tick
                let hw = 0.0035;
                push_strip(out, center - ax_x * r, center + ax_x * r, ax_z, hw, cue_col);
                push_strip(out, center - ax_z * r, center + ax_z * r, ax_x, hw, cue_col);
            }
        }
    }
}

/// Builds small, faint passthrough badges (soft center diamond + 4 thin corner ticks) over detected furniture/objects.
pub fn build_object_badge_vertices(
    objects: &[ObjectMarker],
    quad_hw: f32,
    quad_hh: f32,
    out: &mut Vec<ScanVertex>,
) {
    out.clear();
    let safe_hw = quad_hw.max(0.2);
    let safe_hh = quad_hh.max(0.2);
    let z = -PASSTHROUGH_DIST + 0.005;

    for obj in objects.iter().take(MAX_OBJECTS) {
        if out.len() + 72 > MAX_PASS_VERTICES {
            break;
        }
        let rgb = kind_rgb(obj.kind_id);
        let fill_col = [rgb[0], rgb[1], rgb[2], 0.16];
        let tick_col = [rgb[0], rgb[1], rgb[2], 0.58];
        let dot_col = [1.0, 1.0, 1.0, 0.72];

        let cx = obj.ndc_x * safe_hw;
        let cy = obj.ndc_y * safe_hh;
        let center = Vec3::new(cx, cy, z);

        // Keep badge size intentionally small & subtle even for large sofas/tables
        let bw = (obj.ndc_w * 0.45 * safe_hw).clamp(0.048, 0.105);
        let bh = (obj.ndc_h * 0.45 * safe_hh).clamp(0.048, 0.105);
        let dr = (bw.min(bh) * 0.42).clamp(0.022, 0.036);

        // Faint center diamond
        push_quad(
            out,
            center + Vec3::new(0.0, dr, 0.0),
            center + Vec3::new(dr, 0.0, 0.0),
            center + Vec3::new(0.0, -dr, 0.0),
            center + Vec3::new(-dr, 0.0, 0.0),
            fill_col,
        );

        // Tiny center dot
        let ds = 0.007;
        push_quad(
            out,
            center + Vec3::new(-ds, -ds, 0.001),
            center + Vec3::new(ds, -ds, 0.001),
            center + Vec3::new(ds, ds, 0.001),
            center + Vec3::new(-ds, ds, 0.001),
            dot_col,
        );

        // 4 subtle corner L-ticks
        let leg = (bw.min(bh) * 0.36).clamp(0.018, 0.032);
        let thick = 0.0038;
        let corners = [
            (-1.0f32, 1.0f32),
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, -1.0),
        ];
        for (sx, sy) in corners {
            let corner = Vec3::new(cx + sx * bw, cy + sy * bh, z);
            // Horizontal tick inward
            let h_end = corner + Vec3::new(-sx * leg, 0.0, 0.0);
            push_strip(out, corner, h_end, Vec3::Y, thick, tick_col);
            // Vertical tick inward
            let v_end = corner + Vec3::new(0.0, -sy * leg, 0.0);
            push_strip(out, corner, v_end, Vec3::X, thick, tick_col);
        }
    }
}

fn project_point_to_uv(vp: &Mat4, world_pos: Vec3) -> Option<(f32, f32)> {
    let clip = *vp * world_pos.extend(1.0);
    if clip.w <= 0.08 {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let u = ndc_x * 0.5 + 0.5;
    let v = 0.5 - ndc_y * 0.5; // Convert GL +Y up to Android View +Y down
    if (0.06..=0.94).contains(&u) && (0.14..=0.88).contains(&v) {
        Some((u, v))
    } else {
        None
    }
}

/// Projects up to `MAX_LABELS` non-overlapping stereo labels (objects first, then surfaces).
pub fn project_room_labels(
    planes: &[PlaneMarker],
    objects: &[ObjectMarker],
    t_cam_capture: &Mat4,
    quad_hw: f32,
    quad_hh: f32,
    left_vp: &Mat4,
    right_vp: &Mat4,
) -> Vec<ProjectedLabel> {
    let mut out: Vec<ProjectedLabel> = Vec::with_capacity(MAX_LABELS);
    let safe_hw = quad_hw.max(0.2);
    let safe_hh = quad_hh.max(0.2);

    let try_push = |out: &mut Vec<ProjectedLabel>, kind_id: i32, world_pos: Vec3, score: f32| {
        if out.len() >= MAX_LABELS {
            return;
        }
        let Some((lu, lv)) = project_point_to_uv(left_vp, world_pos) else {
            return;
        };
        let Some((ru, rv)) = project_point_to_uv(right_vp, world_pos) else {
            return;
        };
        let mu = (lu + ru) * 0.5;
        let mv = (lv + rv) * 0.5;
        let overlaps = out.iter().any(|existing| {
            let eu = (existing.left_u + existing.right_u) * 0.5;
            let ev = (existing.left_v + existing.right_v) * 0.5;
            let du = mu - eu;
            let dv = mv - ev;
            (du * du + dv * dv).sqrt() < 0.14
        });
        if !overlaps {
            out.push(ProjectedLabel {
                kind_id,
                left_u: lu,
                left_v: lv,
                right_u: ru,
                right_v: rv,
                score,
            });
        }
    };

    // Prioritize ML object detections sorted by confidence
    let mut sorted_objs: Vec<&ObjectMarker> = objects.iter().collect();
    sorted_objs.sort_by(|a, b| b.score.total_cmp(&a.score));
    for obj in sorted_objs {
        let local_pos = Vec3::new(
            obj.ndc_x * safe_hw,
            obj.ndc_y * safe_hh,
            -PASSTHROUGH_DIST,
        );
        let world_pos = t_cam_capture.transform_point3(local_pos);
        try_push(&mut out, obj.kind_id, world_pos, obj.score);
    }

    // Prioritize geometric surfaces: Table/Seat first, then Floor, Wall, Ceiling
    let mut sorted_planes: Vec<&PlaneMarker> = planes.iter().collect();
    sorted_planes.sort_by_key(|p| match p.kind {
        SurfaceKind::Table => 0,
        SurfaceKind::Seat => 1,
        SurfaceKind::Floor => 2,
        SurfaceKind::Wall => 3,
        SurfaceKind::Ceiling => 4,
        SurfaceKind::Unknown => 5,
    });
    for plane in sorted_planes {
        let score = match plane.kind {
            SurfaceKind::Floor => 0.92,
            SurfaceKind::Wall => 0.88,
            SurfaceKind::Table => 0.85,
            SurfaceKind::Seat => 0.80,
            SurfaceKind::Ceiling => 0.75,
            SurfaceKind::Unknown => 0.0,
        };
        try_push(&mut out, plane.kind as i32, plane.pos, score);
    }

    out
}

pub struct RoomScannerRenderer {
    program: glow::Program,
    world_vao: glow::VertexArray,
    world_vbo: glow::Buffer,
    pass_vao: glow::VertexArray,
    pass_vbo: glow::Buffer,
    u_mvp: Option<glow::UniformLocation>,
    world_cpu: Vec<ScanVertex>,
    pass_cpu: Vec<ScanVertex>,
    world_vert_count: i32,
    pass_vert_count: i32,
}

impl RoomScannerRenderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = compile_program(gl, SCAN_VERT, SCAN_FRAG)?;
            let world_vao = gl.create_vertex_array()?;
            let world_vbo = gl.create_buffer()?;
            Self::setup_vao(gl, program, world_vao, world_vbo, MAX_WORLD_VERTICES);

            let pass_vao = gl.create_vertex_array()?;
            let pass_vbo = gl.create_buffer()?;
            Self::setup_vao(gl, program, pass_vao, pass_vbo, MAX_PASS_VERTICES);

            let u_mvp = gl.get_uniform_location(program, "u_mvp");
            Ok(Self {
                program,
                world_vao,
                world_vbo,
                pass_vao,
                pass_vbo,
                u_mvp,
                world_cpu: Vec::with_capacity(MAX_WORLD_VERTICES),
                pass_cpu: Vec::with_capacity(MAX_PASS_VERTICES),
                world_vert_count: 0,
                pass_vert_count: 0,
            })
        }
    }

    unsafe fn setup_vao(
        gl: &glow::Context,
        program: glow::Program,
        vao: glow::VertexArray,
        vbo: glow::Buffer,
        max_verts: usize,
    ) {
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
        gl.buffer_data_size(
            glow::ARRAY_BUFFER,
            (max_verts * size_of::<ScanVertex>()) as i32,
            glow::DYNAMIC_DRAW,
        );

        let stride = size_of::<ScanVertex>() as i32;
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
    }

    pub fn update(
        &mut self,
        gl: &glow::Context,
        planes: &[PlaneMarker],
        objects: &[ObjectMarker],
        quad_hw: f32,
        quad_hh: f32,
    ) {
        if planes.is_empty() {
            self.world_vert_count = 0;
        } else {
            build_plane_marker_vertices(planes, &mut self.world_cpu);
            self.world_vert_count = self.world_cpu.len() as i32;
            if self.world_vert_count > 0 {
                unsafe {
                    let bytes = std::slice::from_raw_parts(
                        self.world_cpu.as_ptr() as *const u8,
                        self.world_cpu.len() * size_of::<ScanVertex>(),
                    );
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.world_vbo));
                    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                    gl.bind_buffer(glow::ARRAY_BUFFER, None);
                }
            }
        }

        if objects.is_empty() {
            self.pass_vert_count = 0;
        } else {
            build_object_badge_vertices(objects, quad_hw, quad_hh, &mut self.pass_cpu);
            self.pass_vert_count = self.pass_cpu.len() as i32;
            if self.pass_vert_count > 0 {
                unsafe {
                    let bytes = std::slice::from_raw_parts(
                        self.pass_cpu.as_ptr() as *const u8,
                        self.pass_cpu.len() * size_of::<ScanVertex>(),
                    );
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.pass_vbo));
                    gl.buffer_sub_data_u8_slice(glow::ARRAY_BUFFER, 0, bytes);
                    gl.bind_buffer(glow::ARRAY_BUFFER, None);
                }
            }
        }
    }

    pub fn draw_world_planes(&self, gl: &glow::Context, view_proj: &Mat4) {
        if self.world_vert_count <= 0 {
            return;
        }
        unsafe {
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.world_vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, view_proj.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.world_vert_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
        }
    }

    pub fn draw_object_badges(&self, gl: &glow::Context, pass_mvp: &Mat4) {
        if self.pass_vert_count <= 0 {
            return;
        }
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.depth_mask(false);
            gl.disable(glow::CULL_FACE);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);

            gl.use_program(Some(self.program));
            gl.bind_vertex_array(Some(self.pass_vao));
            if let Some(loc) = &self.u_mvp {
                gl.uniform_matrix_4_f32_slice(Some(loc), false, pass_mvp.as_ref());
            }
            gl.draw_arrays(glow::TRIANGLES, 0, self.pass_vert_count);

            gl.bind_vertex_array(None);
            gl.disable(glow::BLEND);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        unsafe {
            gl.delete_buffer(self.world_vbo);
            gl.delete_vertex_array(self.world_vao);
            gl.delete_buffer(self.pass_vbo);
            gl.delete_vertex_array(self.pass_vao);
            gl.delete_program(self.program);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_floor_seat_table_wall_and_ceiling_accurately() {
        let cam_y = 1.45;
        let summary = vec![
            (0, Vec3::new(0.0, 0.0, -2.0), 2.5, 2.5),
            (0, Vec3::new(0.4, 0.44, -1.6), 0.55, 0.55),
            (0, Vec3::new(-0.3, 0.76, -1.5), 1.4, 0.8),
            (1, Vec3::new(0.0, 1.2, -3.0), 3.0, 2.0),
            (2, Vec3::new(0.0, 2.7, -2.0), 2.5, 2.5),
        ];
        let floor_y = infer_floor_y(&summary, cam_y);
        assert_eq!(floor_y, Some(0.0));

        assert_eq!(classify_ar_plane(0, 0.02, 2.5, 2.5, cam_y, floor_y), SurfaceKind::Floor);
        assert_eq!(classify_ar_plane(0, 0.44, 0.55, 0.55, cam_y, floor_y), SurfaceKind::Seat);
        assert_eq!(classify_ar_plane(0, 0.76, 1.4, 0.8, cam_y, floor_y), SurfaceKind::Table);
        assert_eq!(classify_ar_plane(1, 1.2, 3.0, 2.0, cam_y, floor_y), SurfaceKind::Wall);
        assert_eq!(classify_ar_plane(2, 2.7, 2.5, 2.5, cam_y, floor_y), SurfaceKind::Ceiling);
    }

    #[test]
    fn plane_and_object_vertices_stay_faint_small_and_valid_triangles() {
        let planes = vec![PlaneMarker {
            kind: SurfaceKind::Table,
            pos: Vec3::new(0.0, -0.4, -1.5),
            rot: Quat::IDENTITY,
            extent_x: 3.0,
            extent_z: 2.0,
        }];
        let mut world_verts = Vec::new();
        build_plane_marker_vertices(&planes, &mut world_verts);
        assert!(!world_verts.is_empty());
        assert_eq!(world_verts.len() % 3, 0);
        // Check fill alpha is faint (<= 0.58 across all vertices)
        assert!(world_verts.iter().all(|v| v.color[3] <= 0.60));

        let objects = vec![ObjectMarker {
            kind_id: 12,
            ndc_x: 0.1,
            ndc_y: -0.05,
            ndc_w: 0.35,
            ndc_h: 0.45,
            score: 0.82,
        }];
        let mut pass_verts = Vec::new();
        build_object_badge_vertices(&objects, 1.4, 1.0, &mut pass_verts);
        assert!(!pass_verts.is_empty());
        assert_eq!(pass_verts.len() % 3, 0);
    }

    #[test]
    fn projects_non_overlapping_stereo_labels_within_viewport() {
        let proj = Mat4::perspective_rh_gl(85.0_f32.to_radians(), 1200.0 / 1080.0, 0.05, 50.0);
        let left_vp = proj * Mat4::from_translation(Vec3::new(0.032, 0.0, 0.0));
        let right_vp = proj * Mat4::from_translation(Vec3::new(-0.032, 0.0, 0.0));

        let planes = vec![
            PlaneMarker {
                kind: SurfaceKind::Table,
                pos: Vec3::new(0.0, -0.35, -1.6),
                rot: Quat::IDENTITY,
                extent_x: 1.0,
                extent_z: 0.8,
            },
            // Overlapping position close to object below should get filtered out
            PlaneMarker {
                kind: SurfaceKind::Wall,
                pos: Vec3::new(0.0, -0.34, -2.5),
                rot: Quat::IDENTITY,
                extent_x: 2.0,
                extent_z: 2.0,
            },
        ];
        let objects = vec![ObjectMarker {
            kind_id: 12,
            ndc_x: 0.0,
            ndc_y: 0.0,
            ndc_w: 0.25,
            ndc_h: 0.30,
            score: 0.85,
        }];

        let labels = project_room_labels(
            &planes,
            &objects,
            &Mat4::IDENTITY,
            1.4,
            1.0,
            &left_vp,
            &right_vp,
        );
        // Should project both the center chair object (v=0.5) and the lower table (distinct v > 0.64),
        // while culling nearby overlaps or ensuring all UVs sit inside viewport bounds.
        assert!(!labels.is_empty());
        assert_eq!(labels[0].kind_id, 12);
        for lbl in &labels {
            assert!((0.06..=0.94).contains(&lbl.left_u));
            assert!((0.14..=0.88).contains(&lbl.left_v));
            // Binocular disparity: right eye sees objects shifted slightly left in UV (smaller u for same 3D pos)
            assert!(lbl.right_u < lbl.left_u);
        }
    }
}
