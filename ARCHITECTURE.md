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
| **1-Eye Mono AR / 2-Eye VR Toggle** | Default **1 Mắt** (`stereoMode = false`, zero IPD, full `2400x1080`, `1.02` overscan, `near = 0.02m`) with independent `LayoutParams` per HUD pane; 1-tap toggle switches to **2 Mắt** cardboard stereo (`1.06` overscan) + dynamic `ARSession.setDisplayGeometry`. |
| **Incremental 3D Fusion (`spatial_mesh.rs`)** | **Rule 1 (Single Canonical State):** ARCore true `plane.polygon` merged via Coplanar 2D Convex Hull Union (max `32` planes, adaptive `24cm` ring tessellation) + `192` Sparse Hashed Chunks (`16x16x16` @ **`4cm`** with quantized `0.15mm` sub-voxel centroids, `~3.8 MB` RAM, LRU eviction). **Rule 2 (Static Lock + Old/New Realtime):** Static multi-view accumulation (`>=64` lock across 3+ hits from 5Hz SLAM `PointCloud` + 3Hz high-confidence `RAW_DEPTH_ONLY` `24x18` grid `conf>=175`, `<28` hysteresis) + `3.6cm`-step Free-Space Ray Carving erases moved objects; `InstantPlacement` (`hit_kind=4`) stays preview-only dashed reticle. **Rule 3 & 4 (Smooth Surface Nets + Caps):** FPS-governed **Surface Nets** + 1-pass Laplacian smoothing (`1..2` dirty chunks/tick) drapes a continuous organic triangle skin over static furniture, capped at `54,000` vertices + Frustum/`6m` culling. |
| **Hardware Z-Buffer Mesh Occlusion & `🔺 Lưới △`** | **Pass 0 Depth Pre-Pass (`draw_depth_occlusion`):** Writes static locked room mesh into Hardware Z-Buffer (`color_mask=false`, `-1.6cm` normal bias so items ON tables never clip) -> **automatically hides occluded back-scan mesh layers AND occludes virtual 3D balls/cubes/drawings behind real furniture/walls** with zero raw-depth jitter (toggleable via `⚙ ➔ 🛡 Che khuất`). Mode `4` (`🔺 Lưới △`, default) renders crisp GLES 3.0 `fwidth(v_bary)` triangle sheets in 1 static draw call. |
| **Persistent `.arroom` & 4-DoF Reloc (`room_persistence.rs`)** | **Rule 5 (Compact Persistence):** Atomic async disk I/O (`RoomMapStorage.kt`) saves merged canonical planes, sparse `(idx: u16, weight: u8)` solid voxels (3B/cell), and up to `8` `32x24` grayscale keyframes to `<app>/files/rooms/*.arroom` (~15–150 KB), plus standard Wavefront `.obj` export. **Rule 6 (4-DoF Offline Relocalization):** 1Hz keyframe capture while scanning; 4Hz brightness-invariant shifted-NCC matching (`>=0.72` x2 frames) auto-aligns `(yaw, tx, ty, tz)` + merges saved prior (`weight<=96`, carvable if furniture moved!) into live state, with instant **`📍 Khớp ngay`** 1-tap fallback. |
| **Zero-Copy Camera** | ARCore `CameraConfig` picks `imageSize<=480p` (CPU) + `textureSize>=1080p` (GPU OES `0x8D65`); `DepthMode.DISABLED`; 0x MSAA EGL surface; ungrounded demo cube/octahedron hidden by default (`show_demo_scene = false`). |
| **Full-Screen 1:1 Scale** | Cover-fit `compute_eye_projection` (`tan_v = min(cam_tan_v, cam_tan_h / aspect) / (fov_scale * overscan)`) fills 100% edge-to-edge at `100%` while locking world-to-camera scale 1:1. |
| **120Hz Async Timewarp** | `ArSession.update()` throttled to ~32Hz (`31ms`); `onDrawFrame()` runs at 120Hz (`preferredRefreshRate` + `Surface.setFrameRate`) using `ar_rot * (gyro_at_capture.inv() * gyro_latest)`. |
| **RAM Vision Ring & Lazy AI** | Pre-allocated 2MB 3D YUV->RGBA LUT, 4KB YCbCr skin mask, 4-slot `RwLock` frame ring (`vision_ram.rs`); Hand AI (`handMode = 0` default) and Object AI (`roomScanMode = 2`) lazy-initialize TFLite/GPU delegates only when toggled ON. |
| **Realistic 4x 3D Physics** | World-anchored `anchored_floor_y` (never chases `cam_pos.y`); **`⬇ Thả`** vertical reticle drop; **`🎯 Ném`** gravity-compensated ballistic throw; 4x substeps; exact Table/Seat horizontal edges (`extra = 0.0`); upright flat cube settling + static friction snap; dual-layer contact shadows & expanding impact ripple rings (`physics_3d.rs`). |
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
| `nativeSetStereoMode` | `enabled: Boolean` (`false=Mono, true=Stereo`) | `Unit` |
| `nativeSetMeshOcclusion` | `enabled: Boolean` (Hardware Z-Buffer static mesh occlusion) | `Unit` |
| `nativeSetDrawMode` | `drawMode: Int` | `Unit` |
| `nativeClearDrawings` | None | `Unit` |
| `nativeRecenter` | None | `Unit` |
| `nativeOnHandLandmarks` | `handCount: Int, packedData: FloatArray(210)` | `Int` (`0..8`, `101..105`) |
| `nativeSetRoomScanMode` | `mode: Int` (`0=Off, 1=Lưới 3D, 2=+AI Vật, 3=Ẩn lưới, 4=Lưới △`) | `Unit` |
| `nativeOnArPlanes` | `planeCount: Int, packedPlanes: FloatArray(120)` | `Unit` |
| `nativeOnArPlanePolygons` | `planeCount: Int, packedMeta(160), polyCounts(16), packedPolyXz(768)` | `Unit` |
| `nativeOnDetectedObjects` | `objectCount: Int, packedObjects: FloatArray(36), poseAndTan: FloatArray(9)` | `Unit` |
| `nativeOnPointCloud` | `pointCount: Int, packedXyzc: FloatArray(2048)` | `Unit` |
| `nativeOnSurfaceReticle` | `valid, hitKind (1=Plane,2=Point,3=Floor,4=Instant), px..nz, dist, outInfo(3)` | `Unit` |
| `nativeSpawnPhysicsBody` | `throwSpeed: Float` | `Unit` |
| `nativeDropPhysicsAtReticle` | None | `Unit` |
| `nativeThrowPhysicsAtReticle` | `throwSpeed: Float` | `Unit` |
| `nativeClearPhysicsBodies` | None | `Unit` |
| `nativeGetRoomLabels` | `outArr: FloatArray(36)` | `Int` (`0..6`) |
| `nativeGetPerfMetrics` | `outArr: FloatArray(21)` (`+chunks, +tris, +kfs, +relocState, +relocPct`) | `Unit` |
| `nativeGetRenderFrameMs` | None | `Float` |
| `nativeIngestCameraImage` | `yBuf, uBuf, vBuf, yLen, uLen, vLen, w, h, strides, rotDeg, ndcBasis` | `Long` (`seq`) |
| `nativeConvertLatestFrameRgba` | `outRgbaBuf, dstW, dstH, lastSeq, updateMotion, outMeta(19)` | `Long` (`seq`) |
| `nativeTickRelocalizer` | None (1Hz scan / 4Hz reloc NCC tick) | `Int` (`0=idle,1=match,2=auto,3=manual`) |
| `nativeSerializeRoom` | None | `ByteArray?` (`.arroom` v1) |
| `nativeLoadRoomBytes` | `data: ByteArray, outCounts: IntArray(3)` | `Boolean` |
| `nativeForceRelocalizeNow` | None | `Boolean` |
| `nativeCancelRelocalize` | None | `Unit` |
| `nativeExportRoomObj` | None | `String` (Wavefront `.obj`) |
| `nativeDrawFrame` | None | `Unit` |
