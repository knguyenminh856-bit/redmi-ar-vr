# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Update `ARCHITECTURE.md`, `TASKS.md`, `README.md`, `.gitignore`, `build.gradle.kts` | [x] | Documented full-screen 2.5D voxels & rigid-body physics |
| 2. Rust Core | Add `physics_3d.rs`, upgrade `room_scanner.rs` (6cm PointCloud voxels + 2.5D AABBs), cover-fit 100% fullscreen projection (`renderer.rs`, `lib.rs`) | [x] | 23 unit tests + `aarch64-linux-android` pass |
| 3. Android Shell | Add `acquirePointCloud` (5Hz), `🎲 Thả` button, `KEY_FOV_PCT_V2=100`, 55..105% slider, compact toolbar & telemetry | [x] | Compiles cleanly (`:app:assembleDebug`) |
| 4. CI/CD APK | GitHub Actions `build-apk.yml` (`arm64-v8a` `.so` + signed `.apk`) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Verify tests & Gradle build, clean, commit, push `main` | [x] | Triggered Build #7 on `latest-apk` |
| 6. Build #8 3D Realism | 1-Eye Mono default + 2-Eye toggle, live ARCore HitTest reticle + patch synthesis, world-locked 20cm metric grid, anchored floor, 4x substeps + exact table edges + cube upright settling + shadows & ripples + lazy Hand/Object AI | [x] | 28 Rust tests + `:app:assembleDebug` pass |
