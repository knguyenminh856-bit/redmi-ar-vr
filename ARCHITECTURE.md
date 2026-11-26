# Architecture — Redmi Note 14 4G AR/VR Passthrough (Rust + GLES 3.0)

## 1. Overview & Target Hardware
- **Device:** Xiaomi Redmi Note 14 4G (`arm64-v8a`, MediaTek Helio G99-Ultra, Mali-G57 MC2 GPU, 6.67" AMOLED `2400x1080` @ 120Hz).
- **Goal:** Ultra-smooth 120Hz stereo passthrough + 6DoF 3D objects + fast GPU MediaPipe 5-finger tracking, combos, smooth 3D air-drawing & OTA self-updating.
- **Stack:** Thin Android Kotlin shell (`MainActivity.kt`, `HandTrackerHelper.kt`, `RoomMapStorage.kt`, `AppUpdater.kt`, `ApkFileProvider.kt`, `NativeBridge.kt`) + Pure Rust core (`glow` OpenGL ES 3.0, `glam`, `jni`). (Broken TFLite Object AI `RoomScannerHelper.kt` & `efficientdet_lite0.tflite` removed.)

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
| **Adaptive `2.5cm` Multi-Res TSDF, 360° Memory & Flat Merge (`spatial_mesh.rs`)** | **Full-Room 360° Memory:** `1,200` Sparse Chunks (`~19.2 MB` RAM, `CELL_SIZE = 2.5cm`, `±7.5cm` band) + `150,000`-vertex (`50k` tris) full-360° GPU VBO (ZERO behind-camera VBO stripping!). **Protected Memory & Turn-Back Anti-Churn:** Confirmed surfaces (`old_w >= 46`) resist drift (`0.28×` redundant weight) and require `2` consecutive high-confidence (`c0 >= 85`, `9.5cm` behind) frames before carving, paired with a `240ms` post-turn depth settle gate (`0.90 rad/s`) + central `80%` border crop (`dw/10`, `dh/12`); fresh glitches (`< 46`) still erase in 1 frame + BFS floater pruner (`< 7` voxels). **Real Scanned Floors + Coplanar Flat Merging:** ARCore `PlaneFindingMode.DISABLED`; TSDF scans floors, walls, and tabletops natively and collapses flat faces (`curv < 0.085`) into 1 crisp sheet per chunk, while close-up bumpy detail (`<= 0.95m`) unlocks `2.5cm` amber-gold micro-triangles (`> 0.95m` coalesces to `5cm` emerald). |
| **Auto-Cached Phone Persistence (`RoomMapStorage.kt`, `room_persistence.rs`)** | Internal `filesDir/rooms/auto_room_cache.arroom` (`.arroom` v2 TSDF up to `2,048` chunks + `16` visual keyframes spaced `26°`/45cm, rotation gap-fill + full-map meshing on reload): **auto-saves** on `🔒 Khóa`, `onPause()`, and every `20s`; **auto-loads** on app launch with 4-DoF relocalization (`📍 Khớp ngay` + **`🧹 Xóa cache`** on top bar, reloc bar, `💾 Phòng`, and `⚙ Cài đặt`). |
| **Streamlined Large-Button UI (`MainActivity.kt`)** | Dead `scene_3d.rs` & floating room labels removed. Top bar trimmed to 7 thumb-friendly bold buttons (`11.5sp`, `minHeight = 38dp`: `🔺 Lưới`, `⚡ Quét mạnh/🔒 Khóa`, `🔦 Đèn`, `🧹 Xóa cache`, `💾 Phòng`, `🔄/⬇`, `⚙ Cài đặt`); bottom-right dock (`🗑 | ⬇ Thả | 🎯 Ném`, `13.5sp`, `46dp`); secondary toggles (`1/2 Mắt`, `AI Tay`, `Vẽ`, `Thông số`, `Che khuất`) inside spacious `⚙ Cài đặt` drawer. |
| **Hardware Z-Buffer Mesh Occlusion & `🔺 Lưới △`** | **Pass 0 Depth Pre-Pass (`draw_depth_occlusion`):** Writes static locked conformal TSDF mesh into Hardware Z-Buffer (`color_mask=false`, `-1.4cm` normal bias so items ON tables never clip) -> **automatically hides occluded back-scan mesh layers AND occludes virtual 3D balls/cubes/drawings behind real objects/furniture/walls** at 120Hz (toggleable via `⚙ ➔ 🛡 Che khuất`). Mode `4` (`🔺 Lưới △`, default) renders surface-normal color-coded GLES 3.0 `fwidth(v_bary)` triangles with back-face culling. |
| **Persistent `.arroom` v2 & 4-DoF Reloc (`room_persistence.rs`)** | **Compact TSDF Persistence:** Saves canonical planes, sparse narrow-band TSDF `(idx: u16, tsdf: i8, weight: u8)` (4B/cell, v1 backward-compatible), and up to `8` `32x24` grayscale keyframes to `<app>/files/rooms/*.arroom`, plus Wavefront `.obj` export. Auto-switches loaded rooms into **`🛡 Duy trì`** (Compare & Maintain) mode immediately after 4-DoF relocalization! |
| **Zero-Copy Camera & Depth** | ARCore `CameraConfig` picks `imageSize<=480p` (CPU) + `textureSize>=1080p` (GPU OES `0x8D65`); `DepthMode.AUTOMATIC` (fallback `RAW_DEPTH_ONLY` -> `DISABLED`); 0x MSAA EGL surface; direct `ByteBuffer` zero-copy depth ingestion into Rust. |
| **Full-Screen 1:1 Scale** | Cover-fit `compute_eye_projection` (`tan_v = min(cam_tan_v, cam_tan_h / aspect) / (fov_scale * overscan)`) fills 100% edge-to-edge at `100%` while locking world-to-camera scale 1:1. |
| **120Hz Async Timewarp** | `ArSession.update()` throttled to ~32Hz (`31ms`); `onDrawFrame()` runs at 120Hz (`preferredRefreshRate` + `Surface.setFrameRate`) using `ar_rot * (gyro_at_capture.inv() * gyro_latest)`. |
| **RAM Vision Ring & Lazy AI** | Pre-allocated 2MB 3D YUV->RGBA LUT, 4KB YCbCr skin mask, 4-slot `RwLock` frame ring (`vision_ram.rs`); Hand AI (`handMode = 0` default) and Object AI (`roomScanMode = 2`) lazy-initialize TFLite/GPU delegates only when toggled ON. |
| **Scanned-Only Reality (`physics_3d.rs`, `spatial_mesh.rs`)** | **Zero Auto-Generated Planes:** Deleted `ensure_explicit_drop_patch`, fake `fallback_floor_y` auto-floor, ARCore `Plane`/`InstantPlacement` hit-testing & colliders. Both the **120Hz aiming reticle** (`raycast_scanned_surface`) and **4x rigid-body physics & contact shadows** (`sample_scanned_collision`, trilinear `2.5cm` TSDF field) interact **100% exclusively** with actually scanned TSDF mesh (`total_tris_len() > 0 && weight >= MESH_MIN_WEIGHT`). Unscanned space is true empty air: items fall freely with no shadow and auto-despawn below `cam_pos.y - 5.0m`. |
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
| `nativeSetScanTurbo` | `enabled: Boolean` (`true=⚡ 9Hz Turbo, false=🔒 2Hz True Lock`) | `Unit` |
| `nativeClearSpatialMesh` | None (wipes TSDF mesh/planes/keyframes cleanly for fresh rescan) | `Unit` |
| `nativeIngestDepthMap` | `depthBuf, dims/strides, opt confBuf, isRawOnly, intrinsics, sensorPose` | `Int` (`updated_voxels`) |
| `nativeSetDrawMode` | `drawMode: Int` | `Unit` |
| `nativeClearDrawings` | None | `Unit` |
| `nativeRecenter` | None | `Unit` |
| `nativeOnHandLandmarks` | `handCount: Int, packedData: FloatArray(210)` | `Int` (`0..8`, `101..105`) |
| `nativeSetRoomScanMode` | `mode: Int` (`0=Off, 3=Ẩn lưới, 4=Lưới △`) | `Unit` |
| `nativeOnArPlanePolygons` | `planeCount: Int, packedMeta(160), polyCounts(16), packedPolyXz(768)` | `Unit` |
| `nativeOnPointCloud` | `pointCount: Int, packedXyzc: FloatArray(2048)` | `Unit` |
| `nativeSpawnPhysicsBody` | `throwSpeed: Float` | `Unit` |
| `nativeDropPhysicsAtReticle` | None | `Unit` |
| `nativeThrowPhysicsAtReticle` | `throwSpeed: Float` | `Unit` |
| `nativeClearPhysicsBodies` | None | `Unit` |
| `nativeGetPerfMetrics` | `outArr: FloatArray(26)` (`+chunks, +tris, +kfs, +reloc, +depth, +turbo, +diff, +detailTris, +flatTris`) | `Unit` |
| `nativeGetRenderFrameMs` | None | `Float` |
| `nativeIngestCameraImage` | `yBuf, uBuf, vBuf, yLen, uLen, vLen, w, h, strides, rotDeg, ndcBasis` | `Long` (`seq`) |
| `nativeConvertLatestFrameRgba` | `outRgbaBuf, dstW, dstH, lastSeq, updateMotion, outMeta(19)` | `Long` (`seq`) |
| `nativeTickRelocalizer` | None (1Hz scan / 4Hz reloc NCC tick) | `Int` (`0=idle,1=match,2=auto,3=manual`) |
| `nativeSerializeRoom` | None | `ByteArray?` (`.arroom` v2 TSDF) |
| `nativeLoadRoomBytes` | `data: ByteArray, outCounts: IntArray(3)` | `Boolean` |
| `nativeForceRelocalizeNow` | None | `Boolean` |
| `nativeCancelRelocalize` | None | `Unit` |
| `nativeExportRoomObj` | None | `String` (Wavefront `.obj`) |
| `nativeDrawFrame` | None | `Unit` |
