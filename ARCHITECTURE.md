# Architecture — Redmi Note 14 4G AR/VR Passthrough (Rust + GLES 3.0)

## 1. Overview & Target Hardware
- **Device:** Xiaomi Redmi Note 14 4G (`arm64-v8a`, MediaTek Helio G99-Ultra, Mali-G57 MC2 GPU, 6.67" AMOLED `2400x1080` @ 120Hz).
- **Goal:** Ultra-lightweight (<5 MB APK, <80 MB RAM), zero-GC stereo passthrough + 2 floating 3D objects with minimal motion sickness.
- **Stack:** Thin Android Kotlin shell (permissions, 120Hz display mode, ARCore/Camera2, `TYPE_GAME_ROTATION_VECTOR`) + Pure Rust core (`glow` OpenGL ES 3.0, `glam`, `jni`).

## 2. Directory Layout
| Path | Purpose |
| :--- | :--- |
| `ARCHITECTURE.md` | System architecture, JNI contract, and coordinate math |
| `TASKS.md` | Phased build and verification checklist |
| `README.md` | Setup, VR box calibration, and GitHub Actions APK download guide |
| `rust-core/` | Pure Rust `ar_vr_core` (`cdylib` + `rlib`) stereo GLES renderer & timewarp |
| `android/` | Minimal Gradle Kotlin shell (`MainActivity.kt`, `NativeBridge.kt`) |
| `.github/workflows/build-apk.yml` | Cloud CI compiling `arm64-v8a` `.so` via `cargo-ndk` & packaging `.apk` |

## 3. Anti-Motion-Sickness & Stereo Pipeline
| Mechanism | Specification |
| :--- | :--- |
| **Zero-Copy Camera** | Hardware `SurfaceTexture` bound to `GL_TEXTURE_EXTERNAL_OES` (`0x8D65`) sampled in GLSL ES 1.00 (`samplerExternalOES`). |
| **1:1 Angular Scale** | Quad at virtual distance `2.5m` sized `half_w = 2.5 * tan_half_hfov * fov_scale`, `half_h = 2.5 * tan_half_vfov * fov_scale` over `#000000` AMOLED surround. |
| **120Hz Async Timewarp** | `ArSession.update()` throttled to ~32Hz (`31ms`); `onDrawFrame()` runs at 120Hz using `ar_rot * (gyro_at_capture.inv() * gyro_latest)`. |
| **6DoF Stereo 3D** | Split viewport (`0..W/2` Left, `W/2..W` Right), `IPD = 64mm` (`±0.032m` along head `+X`), `85°` vertical eye FOV, two procedural lit meshes (Cube at `-1.25m`, Octahedron at `-1.65m`). |
| **Hand Tracking & Gestures** | Async MediaPipe `HandLandmarker` (`tasks-vision:1.2.3`) on `320px` YUV worker; Rust draws 21-point 5-finger bone ribbons on passthrough plane + classifies `Fist`, `Pinch`, `FingerGun`, `Peace` with 3-frame debounce -> dual-eye stereo popup. |
| **Automatic Fallback** | Uses ARCore 6DoF when available; seamlessly falls back to `Camera2` + 120Hz 3DoF Gyroscope if Google Play Services for AR is absent. |

## 4. Coordinate Frames & JNI Contract
- **Sensor → OpenGL Head Frame:** `C_sensor_to_gl = Rx(-π/2)`; `Q_head = normalize(C_sensor_to_gl * Q_sensor * display_to_device(display_rotation))`.
- **Thread Separation:** `SharedInputState` (`Mutex`) receives sensor/AR updates from UI/Sensor threads; `GlRenderer` owns `glow::Context` and GL handles exclusively on the GL thread.

| JNI Method (`com.minhkhoi.arvr.NativeBridge`) | Parameters | Return |
| :--- | :--- | :--- |
| `nativeInit()` | None | `Unit` |
| `nativeOnSurfaceCreated()` | None | `Int` (`camera_tex_id`) |
| `nativeOnSurfaceChanged(width, height)` | `Int, Int` | `Unit` |
| `nativeOnGyroRotation(qx, qy, qz, qw, displayRotation)` | `Float x4, Int` | `Unit` |
| `nativeOnArFrame(px, py, pz, rx, ry, rz, rw, tanHalfH, tanHalfV, uvs, validTracking)` | `Float x9, FloatArray(8), Boolean` | `Unit` |
| `nativeOnFallbackCameraFrame(tanHalfH, tanHalfV, uvs)` | `Float, Float, FloatArray(8)` | `Unit` |
| `nativeSetCalibration(fovScale, ipdMeters)` | `Float, Float` | `Unit` |
| `nativeRecenter()` | None | `Unit` |
| `nativeOnHandLandmarks(handCount, packedData)` | `Int, FloatArray(210)` | `Int` (gesture `0..4`) |
| `nativeDrawFrame()` | None | `Unit` |
