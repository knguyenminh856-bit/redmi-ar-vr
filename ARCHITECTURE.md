# Architecture — Redmi Note 14 4G AR/VR Passthrough (Rust + GLES 3.0)

## 1. Overview & Target Hardware
- **Device:** Xiaomi Redmi Note 14 4G (`arm64-v8a`, MediaTek Helio G99-Ultra, Mali-G57 MC2 GPU, 6.67" AMOLED `2400x1080` @ 120Hz).
- **Goal:** Ultra-smooth 120Hz stereo passthrough + 6DoF 3D objects + real-world Depth Occlusion + 1€-smoothed 5-finger tracking, combos & 3D air-drawing.
- **Stack:** Thin Android Kotlin shell (`MainActivity.kt`, `HandTrackerHelper.kt`, `NativeBridge.kt`) + Pure Rust core (`glow` OpenGL ES 3.0, `glam`, `jni`).

## 2. Directory Layout
| Path | Purpose |
| :--- | :--- |
| `ARCHITECTURE.md` | System architecture, JNI contract, depth & combo specs |
| `TASKS.md` | Phased build and verification checklist |
| `README.md` | Setup, VR box calibration, gestures/combos & APK download guide |
| `rust-core/` | Pure Rust `ar_vr_core` (`cdylib` + `rlib`) stereo GLES renderer, depth occlusion & finger canvas |
| `android/` | Minimal Gradle Kotlin shell (ARCore Depth + Camera2 fallback + CPU MediaPipe + Stereo HUD) |
| `.github/workflows/build-apk.yml` | Cloud CI compiling `arm64-v8a` `.so` via `cargo-ndk` & packaging `.apk` |

## 3. Anti-Motion-Sickness, Optimization & XR Pipeline
| Mechanism | Specification |
| :--- | :--- |
| **Zero-Copy Camera** | `CameraConfig` picks `imageSize<=480p` (CPU) + `textureSize>=1080p` (GPU OES `0x8D65`), 0x MSAA EGL surface. |
| **1:1 Angular Scale** | Quad at virtual distance `2.5m` sized `half_w = 2.5 * tan_half_hfov * fov_scale`, `half_h = 2.5 * tan_half_vfov * fov_scale` over `#000000` AMOLED surround. |
| **120Hz Async Timewarp** | `ArSession.update()` throttled to ~32Hz (`31ms`); `onDrawFrame()` runs at 120Hz using `ar_rot * (gyro_at_capture.inv() * gyro_latest)`. |
| **Hybrid Depth Occlusion** | ARCore `DEPTH16` (`15Hz`, `GL_RG8` shader discard) + instant 120Hz hardware Z-prepass (`color_mask=false`) of real hand/fingers silhouette (`depth_occlusion.rs`). |
| **CPU MediaPipe + 1€ Filter** | `Delegate.CPU` on background thread (`224px` direct strided YUV readout, `20Hz`) + 120Hz `OneEuroFilter` in Rust (`hand_tracking.rs`). |
| **Gestures, Combos & Drawing** | 8 single poses + 5 timed 2-step combos (`101..105`: pull/push, boost, recolor, clear draw, recenter) + 3D neon fingertip ribbons (`finger_draw.rs`). |
| **Stereo Telemetry HUD** | Dual-eye synchronized live stats (SoC, Mali GPU, Display Hz, Render/AI FPS, Depth m, Fingertip XYZ, Battery °C, RAM MB). |

## 4. JNI Contract (`com.minhkhoi.arvr.NativeBridge`)
| Method | Parameters | Return |
| :--- | :--- | :--- |
| `nativeInit` | None | `Unit` |
| `nativeOnSurfaceCreated` | None | `Int` (`camera_tex_id`) |
| `nativeOnSurfaceChanged` | `width: Int, height: Int` | `Unit` |
| `nativeGetGpuInfo` | None | `String` |
| `nativeOnGyroRotation` | `qx, qy, qz, qw: Float, displayRotation: Int` | `Unit` |
| `nativeOnArFrame` | `px..rw, tanHalfH, tanHalfV, uvs[8], validTracking` | `Unit` |
| `nativeOnFallbackCameraFrame` | `tanHalfH, tanHalfV, uvs[8]` | `Unit` |
| `nativeOnDepthFrame` | `width, height, bytes, d0x..dyy` | `Unit` |
| `nativeSetCalibration` | `fovScale, ipdMeters` | `Unit` |
| `nativeSetFeatures` | `depthOcclusion: Boolean, drawMode: Int` | `Unit` |
| `nativeClearDrawings` | None | `Unit` |
| `nativeRecenter` | None | `Unit` |
| `nativeOnHandLandmarks` | `handCount: Int, packedData: FloatArray(210)` | `Int` (`0..8`, `101..105`) |
| `nativeGetPerfMetrics` | `outArr: FloatArray(10)` | `Unit` |
| `nativeDrawFrame` | None | `Unit` |
