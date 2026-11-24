# Architecture — Redmi Note 14 4G AR/VR Passthrough (Rust + GLES 3.0)

## 1. Overview & Target Hardware
- **Device:** Xiaomi Redmi Note 14 4G (`arm64-v8a`, MediaTek Helio G99-Ultra, Mali-G57 MC2 GPU, 6.67" AMOLED `2400x1080` @ 120Hz).
- **Goal:** Ultra-smooth 120Hz stereo passthrough + 6DoF 3D objects + fast GPU MediaPipe 5-finger tracking, combos, smooth 3D air-drawing & OTA self-updating.
- **Stack:** Thin Android Kotlin shell (`MainActivity.kt`, `HandTrackerHelper.kt`, `RoomScannerHelper.kt`, `AppUpdater.kt`, `ApkFileProvider.kt`, `NativeBridge.kt`) + Pure Rust core (`glow` OpenGL ES 3.0, `glam`, `jni`).

## 2. Directory Layout
| Path | Purpose |
| :--- | :--- |
| `ARCHITECTURE.md` | System architecture, JNI contract, hand & OTA specs |
| `TASKS.md` | Phased build and verification checklist |
| `README.md` | Setup, VR calibration, gestures/combos & OTA update guide |
| `rust-core/` | Pure Rust `ar_vr_core` (`cdylib` + `rlib`) stereo GLES renderer & finger canvas |
| `android/` | Gradle Kotlin shell (`debug-fixed.keystore`, MediaPipe GPU, OTA updater, Stereo HUD) |
| `.github/workflows/build-apk.yml` | Cloud CI compiling `arm64-v8a` `.so`, signing `.apk`, and tagging `BuildCommit` |

## 3. XR Pipeline & Self-Update
| Mechanism | Specification |
| :--- | :--- |
| **Zero-Copy Camera** | ARCore `CameraConfig` picks `imageSize<=480p` (CPU) + `textureSize>=1080p` (GPU OES `0x8D65`); `DepthMode.DISABLED`; 0x MSAA EGL surface. |
| **1:1 Angular Scale** | Quad at virtual distance `2.5m` sized `half_w = 2.5 * tan_half_hfov * fov_scale`, `half_h = 2.5 * tan_half_vfov * fov_scale` over `#000000` AMOLED surround. |
| **120Hz Async Timewarp** | `ArSession.update()` throttled to ~32Hz (`31ms`); `onDrawFrame()` runs at 120Hz using `ar_rot * (gyro_at_capture.inv() * gyro_latest)`. |
| **Fast 480p GPU Hand AI** | `Delegate.GPU` (fallback `CPU`) on `480px`-wide `Bitmap` with arrival-time velocity-adaptive smoothing (`hand_tracking.rs`). |
| **Gestures, Combos & Drawing** | 8 single poses + 5 timed combos (`101..105`) + quadratic Bezier dual-layer neon strokes pinned at `0.65m` (`finger_draw.rs`). |
| **Room Surface & Object Scan** | ARCore `PlaneFindingMode.HORIZONTAL_AND_VERTICAL` at `4Hz` (Floor, Wall, Table, Seat, Ceiling) + MediaPipe `ObjectDetector` (`efficientdet_lite0.tflite` INT8, CPU delegate, `320px`, `~2.6Hz`), small faint patches/badges (`room_scanner.rs`) & stereo HUD mini-tags (`6` max). |
| **OTA Self-Updater** | Fixed `debug-fixed.keystore`, `BuildConfig.GIT_COMMIT_SHA` check vs GitHub Release `latest-apk`, streaming download + `ApkFileProvider` install intent. |

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
| `nativeSetCalibration` | `fovScale, ipdMeters` | `Unit` |
| `nativeSetDrawMode` | `drawMode: Int` | `Unit` |
| `nativeClearDrawings` | None | `Unit` |
| `nativeRecenter` | None | `Unit` |
| `nativeOnHandLandmarks` | `handCount: Int, packedData: FloatArray(210)` | `Int` (`0..8`, `101..105`) |
| `nativeSetRoomScanMode` | `mode: Int` (`0=Off, 1=Floor/Wall, 2=Full`) | `Unit` |
| `nativeOnArPlanes` | `planeCount: Int, packedPlanes: FloatArray(120)` | `Unit` |
| `nativeOnDetectedObjects` | `objectCount: Int, packedObjects: FloatArray(36)` | `Unit` |
| `nativeGetRoomLabels` | `outArr: FloatArray(36)` | `Int` (`0..6`) |
| `nativeGetPerfMetrics` | `outArr: FloatArray(10)` | `Unit` |
| `nativeDrawFrame` | None | `Unit` |
