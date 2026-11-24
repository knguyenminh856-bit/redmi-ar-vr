pub mod gyro_predict;
pub mod hand_tracking;
pub mod passthrough;
pub mod renderer;
pub mod scene_3d;

use hand_tracking::{FLOATS_PER_LANDMARK, LANDMARKS_PER_HAND, MAX_HANDS};

use jni::objects::{JClass, JFloatArray};
use jni::sys::{jboolean, jint, JNI_TRUE};
use jni::JNIEnv;
use renderer::{GlRenderer, SharedInputState};
use std::cell::RefCell;
use std::sync::{Mutex, OnceLock};

static INPUT_STATE: OnceLock<Mutex<SharedInputState>> = OnceLock::new();

fn input_state() -> &'static Mutex<SharedInputState> {
    INPUT_STATE.get_or_init(|| Mutex::new(SharedInputState::default()))
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
    let _ = input_state();
    log::info!("Rust AR/VR core initialized");
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

    GL_RENDERER.with(|cell| {
        if let Some(renderer) = cell.borrow_mut().as_mut() {
            renderer.draw_frame(&snap);
        }
    });
}
