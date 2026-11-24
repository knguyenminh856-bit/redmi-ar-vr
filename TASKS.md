# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Update `ARCHITECTURE.md`, `TASKS.md`, `README.md`, `.gitignore`, `build.gradle.kts` | [x] | Documented room scanning + wired `efficientdet_lite0.tflite` |
| 2. Rust Core | Add `vision_ram.rs` (2MB LUT + 4-slot ring), upgrade `room_scanner.rs` (Spatial Cache, wall filter, gravity-aligned markers) & 1:1 `proj_eye` + scissor (`renderer.rs`, `finger_draw.rs`, `lib.rs`) | [x] | 19 unit tests + `aarch64-linux-android` pass |
| 3. Android Shell | Add `ByteBufferImageBuilder` RGBA, pinned DirectByteBuffer assets, `GpuOptions` cache, 3-gear AI governor, `120Hz Surface`, cutout `ALWAYS`, per-eye `setDisplayGeometry`, `🖐 Tay` button, UI `dp()` fix | [x] | Compiles cleanly (`:app:assembleDebug`) |
| 4. CI/CD APK | GitHub Actions `build-apk.yml` (`arm64-v8a` `.so` + signed `.apk`) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Verify tests & Gradle build, clean, commit, push `main` | [x] | Triggered Build #6 on `latest-apk` |
