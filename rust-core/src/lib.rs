pub mod finger_draw;
pub mod gyro_predict;
pub mod hand_tracking;
pub mod passthrough;
pub mod physics_3d;
pub mod renderer;
pub mod room_persistence;
pub mod room_scanner;
pub mod scene_3d;
pub mod spatial_mesh;
pub mod vision_ram;

use glam::{Quat, Vec3};
use hand_tracking::{FLOATS_PER_LANDMARK, LANDMARKS_PER_HAND, MAX_HANDS};
use jni::objects::{JByteArray, JByteBuffer, JClass, JFloatArray, JIntArray};
use jni::sys::{jboolean, jbyteArray, jfloat, jint, jlong, jstring, JNI_FALSE, JNI_TRUE};
use jni::JNIEnv;
use renderer::{GlRenderer, SharedInputState};
use room_scanner::{FLOATS_PER_LABEL, FLOATS_PER_OBJECT, FLOATS_PER_PLANE, MAX_LABELS};
use spatial_mesh::{RawArPlane, MAX_POLY_VERTS};
use vision_ram::{convert_latest_rgba, ingest_camera_planes, init_vision_ram, CaptureMeta};
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

static INPUT_STATE: OnceLock<Mutex<SharedInputState>> = OnceLock::new();
static GPU_INFO: OnceLock<Mutex<String>> = OnceLock::new();

fn input_state() -> &'static Mutex<SharedInputState> {
    INPUT_STATE.get_or_init(|| Mutex::new(SharedInputState::default()))
}

fn gpu_info() -> &'static Mutex<String> {
    GPU_INFO.get_or_init(|| Mutex::new("Unknown GLES3 GPU".to_string()))
}

thread_local! {
    static GL_RENDERER: RefCell<Option<GlRenderer>> = const { RefCell::new(None) };
}

#[cfg(target_os = "android")]
fn init_logging() {
    android_logger::init_once(
        android_logger::Config::default()
            .with_max_level(log::LevelFilter::Info)
            .with_tag("RustARVR"),
    );
}

#[cfg(not(target_os = "android"))]
fn init_logging() {}

#[cfg(target_os = "android")]
fn create_gles_context() -> Result<glow::Context, String> {
    use std::ffi::{c_void, CString};

    type EglGetProcAddressFn = unsafe extern "C" fn(*const libc::c_char) -> *const c_void;

    unsafe {
        let mut gles_lib = libc::dlopen(c"libGLESv3.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        if gles_lib.is_null() {
            gles_lib = libc::dlopen(c"libGLESv2.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        }
        let egl_lib = libc::dlopen(c"libEGL.so".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
        let egl_get_proc: Option<EglGetProcAddressFn> = if !egl_lib.is_null() {
            let sym = libc::dlsym(egl_lib, c"eglGetProcAddress".as_ptr());
            if sym.is_null() {
                None
            } else {
                Some(std::mem::transmute::<*const c_void, EglGetProcAddressFn>(sym))
            }
        } else {
            None
        };

        if gles_lib.is_null() && egl_get_proc.is_none() {
            return Err("Failed to load libGLESv3/libEGL".to_string());
        }

        let gl = glow::Context::from_loader_function(|name| {
            let Ok(cname) = CString::new(name) else {
                return std::ptr::null();
            };
            if !gles_lib.is_null() {
                let ptr = libc::dlsym(gles_lib, cname.as_ptr());
                if !ptr.is_null() {
                    return ptr as *const c_void;
                }
            }
            if let Some(get_proc) = egl_get_proc {
                return get_proc(cname.as_ptr());
            }
            std::ptr::null()
        });
        Ok(gl)
    }
}

#[cfg(not(target_os = "android"))]
fn create_gles_context() -> Result<glow::Context, String> {
    Err("GLES context creation is only supported on Android".to_string())
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeInit(
    _env: JNIEnv,
    _class: JClass,
) {
    init_logging();
    init_vision_ram();
    let _ = input_state();
    log::info!("Rust AR/VR core + 2MB RAM LUT initialized");
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnSurfaceCreated(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    init_logging();
    let tex_id = GL_RENDERER.with(|cell| {
        if let Some(old) = cell.borrow_mut().take() {
            old.destroy();
        }
        match create_gles_context().and_then(GlRenderer::new) {
            Ok(renderer) => {
                let id = renderer.camera_texture_id() as jint;
                if let Ok(mut name) = gpu_info().lock() {
                    *name = renderer.gpu_renderer_name.clone();
                }
                *cell.borrow_mut() = Some(renderer);
                log::info!("GL renderer created, OES texture id={}", id);
                id
            }
            Err(err) => {
                log::error!("Failed to create GL renderer: {}", err);
                0
            }
        }
    });
    tex_id
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnSurfaceChanged(
    _env: JNIEnv,
    _class: JClass,
    width: jint,
    height: jint,
) {
    GL_RENDERER.with(|cell| {
        if let Some(renderer) = cell.borrow_mut().as_mut() {
            renderer.resize(width, height);
        }
    });
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeGetGpuInfo(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let text = gpu_info()
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| "Unknown".to_string());
    match env.new_string(text) {
        Ok(jstr) => jstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnGyroRotation(
    _env: JNIEnv,
    _class: JClass,
    qx: f32,
    qy: f32,
    qz: f32,
    qw: f32,
    display_rotation: jint,
) {
    if let Ok(mut state) = input_state().lock() {
        state.on_gyro_rotation(qx, qy, qz, qw, display_rotation);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnArFrame(
    env: JNIEnv,
    _class: JClass,
    px: f32,
    py: f32,
    pz: f32,
    rx: f32,
    ry: f32,
    rz: f32,
    rw: f32,
    tan_half_h: f32,
    tan_half_v: f32,
    uvs_arr: JFloatArray,
    valid_tracking: jboolean,
) {
    let mut uvs = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    if env.get_float_array_region(&uvs_arr, 0, &mut uvs).is_err() {
        return;
    }
    if let Ok(mut state) = input_state().lock() {
        state.on_ar_frame(
            px,
            py,
            pz,
            rx,
            ry,
            rz,
            rw,
            tan_half_h,
            tan_half_v,
            uvs,
            valid_tracking == JNI_TRUE,
        );
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnFallbackCameraFrame(
    env: JNIEnv,
    _class: JClass,
    tan_half_h: f32,
    tan_half_v: f32,
    uvs_arr: JFloatArray,
) {
    let mut uvs = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    if env.get_float_array_region(&uvs_arr, 0, &mut uvs).is_err() {
        return;
    }
    if let Ok(mut state) = input_state().lock() {
        state.on_fallback_camera_frame(tan_half_h, tan_half_v, uvs);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSetCalibration(
    _env: JNIEnv,
    _class: JClass,
    fov_scale: f32,
    ipd_meters: f32,
) {
    if let Ok(mut state) = input_state().lock() {
        state.set_calibration(fov_scale, ipd_meters);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSetDrawMode(
    _env: JNIEnv,
    _class: JClass,
    draw_mode: jint,
) {
    if let Ok(mut state) = input_state().lock() {
        state.set_draw_mode(draw_mode);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeClearDrawings(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Ok(mut state) = input_state().lock() {
        state.clear_drawings();
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeRecenter(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Ok(mut state) = input_state().lock() {
        state.request_recenter();
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnHandLandmarks(
    env: JNIEnv,
    _class: JClass,
    hand_count: jint,
    packed_arr: JFloatArray,
) -> jint {
    let clamped_count = (hand_count.max(0) as usize).min(MAX_HANDS);
    let expected_len = clamped_count * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK;
    let mut buf = [0.0f32; MAX_HANDS * LANDMARKS_PER_HAND * FLOATS_PER_LANDMARK];
    if expected_len > 0
        && env
            .get_float_array_region(&packed_arr, 0, &mut buf[..expected_len])
            .is_err()
    {
        return 0;
    }

    if let Ok(mut state) = input_state().lock() {
        state.on_hand_landmarks(clamped_count, &buf[..expected_len])
    } else {
        0
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeGetPerfMetrics(
    env: JNIEnv,
    _class: JClass,
    out_arr: JFloatArray,
) {
    let perf = {
        let Ok(state) = input_state().lock() else {
            return;
        };
        state.perf
    };
    let buf = [
        perf.render_fps,
        perf.frame_ms,
        perf.active_hands as f32,
        perf.current_gesture as f32,
        perf.stroke_vertices as f32,
        perf.index_tip_cam[0],
        perf.index_tip_cam[1],
        perf.index_tip_cam[2],
        perf.active_planes as f32,
        perf.active_objects as f32,
        perf.cached_planes as f32,
        perf.voxel_count as f32,
        perf.physics_bodies as f32,
        perf.reticle_hit_kind as f32,
        perf.reticle_surface_kind as f32,
        perf.reticle_dist,
        perf.mesh_chunks as f32,
        perf.mesh_triangles as f32,
        perf.keyframe_count as f32,
        perf.reloc_status as f32,
        perf.reloc_best_pct as f32,
    ];
    let copy_len = env
        .get_array_length(&out_arr)
        .map(|n| (n.max(0) as usize).min(buf.len()))
        .unwrap_or(18);
    let _ = env.set_float_array_region(&out_arr, 0, &buf[..copy_len]);
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeGetRenderFrameMs(
    _env: JNIEnv,
    _class: JClass,
) -> jfloat {
    input_state()
        .lock()
        .map(|s| s.perf.frame_ms)
        .unwrap_or(0.0)
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeIngestCameraImage(
    env: JNIEnv,
    _class: JClass,
    y_buf: JByteBuffer,
    u_buf: JByteBuffer,
    v_buf: JByteBuffer,
    y_len: jint,
    u_len: jint,
    v_len: jint,
    width: jint,
    height: jint,
    y_row_stride: jint,
    y_pixel_stride: jint,
    uv_row_stride: jint,
    uv_pixel_stride: jint,
    rot_deg: jint,
    ndc_basis_arr: JFloatArray,
) -> jlong {
    if width < 2 || height < 2 || y_len <= 0 || u_len <= 0 || v_len <= 0 {
        return 0;
    }
    let mut ndc_basis = [0.0f32; 6];
    if env.get_float_array_region(&ndc_basis_arr, 0, &mut ndc_basis).is_err() {
        return 0;
    }

    let (cam_pos, cam_rot, tan_h, tan_v) = {
        let Ok(state) = input_state().lock() else {
            return 0;
        };
        let (pos, rot) = if state.has_ar_pose {
            (state.ar_pos, state.ar_rot)
        } else {
            let head = if state.has_gyro {
                state.gyro_latest
            } else {
                Quat::IDENTITY
            };
            (Vec3::ZERO, head)
        };
        (
            pos.to_array(),
            rot.to_array(),
            state.cam_tan_half_h,
            state.cam_tan_half_v,
        )
    };

    let (Ok(y_ptr), Ok(y_cap)) = (
        env.get_direct_buffer_address(&y_buf),
        env.get_direct_buffer_capacity(&y_buf),
    ) else {
        return 0;
    };
    let (Ok(u_ptr), Ok(u_cap)) = (
        env.get_direct_buffer_address(&u_buf),
        env.get_direct_buffer_capacity(&u_buf),
    ) else {
        return 0;
    };
    let (Ok(v_ptr), Ok(v_cap)) = (
        env.get_direct_buffer_address(&v_buf),
        env.get_direct_buffer_capacity(&v_buf),
    ) else {
        return 0;
    };

    if y_ptr.is_null() || u_ptr.is_null() || v_ptr.is_null() {
        return 0;
    }

    let y_slice = unsafe { std::slice::from_raw_parts(y_ptr, (y_len as usize).min(y_cap)) };
    let u_slice = unsafe { std::slice::from_raw_parts(u_ptr, (u_len as usize).min(u_cap)) };
    let v_slice = unsafe { std::slice::from_raw_parts(v_ptr, (v_len as usize).min(v_cap)) };

    let seq = ingest_camera_planes(
        y_slice,
        u_slice,
        v_slice,
        width as usize,
        height as usize,
        y_row_stride.max(1) as usize,
        y_pixel_stride.max(1) as usize,
        uv_row_stride.max(1) as usize,
        uv_pixel_stride.max(1) as usize,
        CaptureMeta {
            seq: 0,
            cam_pos,
            cam_rot,
            tan_h,
            tan_v,
            ndc_basis,
            rot_deg,
        },
    );
    seq as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeConvertLatestFrameRgba(
    env: JNIEnv,
    _class: JClass,
    out_rgba_buf: JByteBuffer,
    dst_w: jint,
    dst_h: jint,
    last_seq: jlong,
    update_motion: jboolean,
    out_meta_arr: JFloatArray,
) -> jlong {
    if dst_w < 2 || dst_h < 2 {
        return 0;
    }
    let needed = (dst_w as usize) * (dst_h as usize) * 4;
    let (Ok(out_ptr), Ok(out_cap)) = (
        env.get_direct_buffer_address(&out_rgba_buf),
        env.get_direct_buffer_capacity(&out_rgba_buf),
    ) else {
        return 0;
    };
    if out_ptr.is_null() || out_cap < needed {
        return 0;
    }

    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr, needed) };
    let Some((meta, skin_pm, motion_pm, convert_us)) = convert_latest_rgba(
        out_slice,
        dst_w as usize,
        dst_h as usize,
        last_seq.max(0) as u64,
        update_motion == JNI_TRUE,
    ) else {
        return 0;
    };

    let meta_floats = [
        skin_pm as f32,
        motion_pm as f32,
        convert_us as f32,
        meta.cam_pos[0],
        meta.cam_pos[1],
        meta.cam_pos[2],
        meta.cam_rot[0],
        meta.cam_rot[1],
        meta.cam_rot[2],
        meta.cam_rot[3],
        meta.tan_h,
        meta.tan_v,
        meta.ndc_basis[0],
        meta.ndc_basis[1],
        meta.ndc_basis[2],
        meta.ndc_basis[3],
        meta.ndc_basis[4],
        meta.ndc_basis[5],
        meta.rot_deg as f32,
    ];
    let _ = env.set_float_array_region(&out_meta_arr, 0, &meta_floats);
    meta.seq as jlong
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSetRoomScanMode(
    _env: JNIEnv,
    _class: JClass,
    mode: jint,
) {
    if let Ok(mut state) = input_state().lock() {
        state.set_room_scan_mode(mode);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnArPlanes(
    env: JNIEnv,
    _class: JClass,
    plane_count: jint,
    packed_planes: JFloatArray,
) {
    if plane_count <= 0 {
        if let Ok(mut state) = input_state().lock() {
            state.on_ar_planes(0, &[]);
        }
        return;
    }
    let count = (plane_count as usize).min(room_scanner::MAX_PLANES);
    let mut buf = vec![0.0f32; count * FLOATS_PER_PLANE];
    if env.get_float_array_region(&packed_planes, 0, &mut buf).is_err() {
        return;
    }
    if let Ok(mut state) = input_state().lock() {
        state.on_ar_planes(count, &buf);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnArPlanePolygons(
    env: JNIEnv,
    _class: JClass,
    plane_count: jint,
    packed_meta: JFloatArray,
    poly_counts: JIntArray,
    packed_poly_xz: JFloatArray,
) {
    if plane_count <= 0 {
        return;
    }
    let count = (plane_count as usize).min(16);
    let mut meta = vec![0.0f32; count * FLOATS_PER_PLANE];
    let mut counts = vec![0i32; count];
    if env.get_float_array_region(&packed_meta, 0, &mut meta).is_err()
        || env.get_int_array_region(&poly_counts, 0, &mut counts).is_err()
    {
        return;
    }

    let total_verts: usize = counts
        .iter()
        .map(|&c| (c.max(0) as usize).min(MAX_POLY_VERTS))
        .sum();
    let mut poly_floats = vec![0.0f32; total_verts * 2];
    if total_verts > 0
        && env
            .get_float_array_region(&packed_poly_xz, 0, &mut poly_floats)
            .is_err()
    {
        return;
    }

    let mut raw_planes = Vec::with_capacity(count);
    let mut cursor = 0usize;
    for i in 0..count {
        let b = i * FLOATS_PER_PLANE;
        let v_cnt = (counts[i].max(0) as usize).min(MAX_POLY_VERTS);
        let mut poly_xz = Vec::with_capacity(v_cnt);
        for _ in 0..v_cnt {
            if cursor + 2 <= poly_floats.len() {
                poly_xz.push([poly_floats[cursor], poly_floats[cursor + 1]]);
            }
            cursor += 2;
        }
        raw_planes.push(RawArPlane {
            raw_type: meta[b].round() as i32,
            pos: Vec3::new(meta[b + 1], meta[b + 2], meta[b + 3]),
            rot: Quat::from_xyzw(meta[b + 4], meta[b + 5], meta[b + 6], meta[b + 7]),
            extent_x: meta[b + 8].clamp(0.05, 8.0),
            extent_z: meta[b + 9].clamp(0.05, 8.0),
            polygon_xz: poly_xz,
        });
    }

    if let Ok(mut state) = input_state().lock() {
        state.on_ar_plane_polygons(raw_planes);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnDetectedObjects(
    env: JNIEnv,
    _class: JClass,
    object_count: jint,
    packed_objects: JFloatArray,
    pose_and_tan_arr: JFloatArray,
) {
    let mut pose_buf = [0.0f32; 9];
    if env.get_float_array_region(&pose_and_tan_arr, 0, &mut pose_buf).is_err() {
        return;
    }
    let cam_pos = Vec3::new(pose_buf[0], pose_buf[1], pose_buf[2]);
    let q = Quat::from_xyzw(pose_buf[3], pose_buf[4], pose_buf[5], pose_buf[6]);
    let cam_rot = if q.length_squared() > 1e-6 {
        q.normalize()
    } else {
        Quat::IDENTITY
    };
    let tan_h = pose_buf[7];
    let tan_v = pose_buf[8];

    if object_count <= 0 {
        if let Ok(mut state) = input_state().lock() {
            state.on_detected_objects(0, &[], cam_pos, cam_rot, tan_h, tan_v);
        }
        return;
    }
    let count = (object_count as usize).min(room_scanner::MAX_OBJECTS);
    let mut buf = vec![0.0f32; count * FLOATS_PER_OBJECT];
    if env
        .get_float_array_region(&packed_objects, 0, &mut buf)
        .is_err()
    {
        return;
    }
    if let Ok(mut state) = input_state().lock() {
        state.on_detected_objects(count, &buf, cam_pos, cam_rot, tan_h, tan_v);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnPointCloud(
    env: JNIEnv,
    _class: JClass,
    point_count: jint,
    packed_xyzc: JFloatArray,
) {
    if point_count <= 0 {
        return;
    }
    let count = (point_count as usize).min(room_scanner::MAX_POINT_CLOUD_PTS);
    let mut buf = vec![0.0f32; count * 4];
    if env.get_float_array_region(&packed_xyzc, 0, &mut buf).is_err() {
        return;
    }
    if let Ok(mut state) = input_state().lock() {
        state.on_point_cloud(count, &buf);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSpawnPhysicsBody(
    _env: JNIEnv,
    _class: JClass,
    throw_speed: jfloat,
) {
    if let Ok(mut state) = input_state().lock() {
        state.spawn_physics_from_camera(throw_speed);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSetStereoMode(
    _env: JNIEnv,
    _class: JClass,
    enabled: jboolean,
) {
    if let Ok(mut state) = input_state().lock() {
        state.set_stereo_mode(enabled != 0);
    }
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeOnSurfaceReticle(
    env: JNIEnv,
    _class: JClass,
    valid: jboolean,
    hit_kind: jint,
    px: jfloat,
    py: jfloat,
    pz: jfloat,
    nx: jfloat,
    ny: jfloat,
    nz: jfloat,
    dist: jfloat,
    out_arr: JFloatArray,
) {
    let res = if let Ok(mut state) = input_state().lock() {
        state.on_surface_reticle(
            valid != 0,
            hit_kind,
            Vec3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
            dist,
        )
    } else {
        (0, 0, 0.0)
    };
    let out = [res.0 as f32, res.1 as f32, res.2];
    let _ = env.set_float_array_region(&out_arr, 0, &out);
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeDropPhysicsAtReticle(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Ok(mut state) = input_state().lock() {
        state.drop_physics_at_reticle();
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeThrowPhysicsAtReticle(
    _env: JNIEnv,
    _class: JClass,
    throw_speed: jfloat,
) {
    if let Ok(mut state) = input_state().lock() {
        state.throw_physics_at_reticle(throw_speed);
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeClearPhysicsBodies(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Ok(mut state) = input_state().lock() {
        state.clear_physics_bodies();
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeGetRoomLabels(
    env: JNIEnv,
    _class: JClass,
    out_arr: JFloatArray,
) -> jint {
    let labels = {
        let Ok(state) = input_state().lock() else {
            return 0;
        };
        state.projected_labels.clone()
    };
    let count = labels.len().min(MAX_LABELS);
    if count == 0 {
        return 0;
    }
    let mut buf = [0.0f32; MAX_LABELS * FLOATS_PER_LABEL];
    for (i, lbl) in labels.iter().take(count).enumerate() {
        let base = i * FLOATS_PER_LABEL;
        buf[base] = lbl.kind_id as f32;
        buf[base + 1] = lbl.left_u;
        buf[base + 2] = lbl.left_v;
        buf[base + 3] = lbl.right_u;
        buf[base + 4] = lbl.right_v;
        buf[base + 5] = lbl.score;
    }
    let _ = env.set_float_array_region(&out_arr, 0, &buf[..count * FLOATS_PER_LABEL]);
    count as jint
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeTickRelocalizer(
    _env: JNIEnv,
    _class: JClass,
) -> jint {
    if let Ok(mut state) = input_state().lock() {
        state.poll_visual_keyframes_and_reloc()
    } else {
        0
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeSerializeRoom(
    env: JNIEnv,
    _class: JClass,
) -> jbyteArray {
    let bytes = {
        let Ok(state) = input_state().lock() else {
            return std::ptr::null_mut();
        };
        state.serialize_current_room()
    };
    match env.byte_array_from_slice(&bytes) {
        Ok(arr) => arr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeLoadRoomBytes(
    env: JNIEnv,
    _class: JClass,
    data: JByteArray,
    out_counts: JIntArray,
) -> jboolean {
    let Ok(bytes) = env.convert_byte_array(data) else {
        return JNI_FALSE;
    };
    let result = if let Ok(mut state) = input_state().lock() {
        state.load_room_bytes(&bytes)
    } else {
        return JNI_FALSE;
    };
    match result {
        Ok((planes, voxels, kfs)) => {
            let counts = [planes as i32, voxels as i32, kfs as i32];
            let _ = env.set_int_array_region(&out_counts, 0, &counts);
            JNI_TRUE
        }
        Err(_) => JNI_FALSE,
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeForceRelocalizeNow(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    let locked = if let Ok(mut state) = input_state().lock() {
        state.force_relocalize_now()
    } else {
        false
    };
    if locked { JNI_TRUE } else { JNI_FALSE }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeCancelRelocalize(
    _env: JNIEnv,
    _class: JClass,
) {
    if let Ok(mut state) = input_state().lock() {
        state.cancel_pending_relocalize();
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeExportRoomObj(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let obj_text = {
        let Ok(mut state) = input_state().lock() else {
            return std::ptr::null_mut();
        };
        state.export_current_room_obj()
    };
    match env.new_string(obj_text) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "system" fn Java_com_minhkhoi_arvr_NativeBridge_nativeDrawFrame(
    _env: JNIEnv,
    _class: JClass,
) {
    let snap = {
        let Ok(mut state) = input_state().lock() else {
            return;
        };
        state.snapshot_frame()
    };

    let timing = GL_RENDERER.with(|cell| {
        cell.borrow_mut()
            .as_mut()
            .map(|renderer| renderer.draw_frame(&snap))
    });

    if let Some((fps, frame_ms, labels)) = timing {
        if let Ok(mut state) = input_state().lock() {
            state.record_render_timing(fps, frame_ms);
            state.set_projected_labels(labels);
        }
    }
}
