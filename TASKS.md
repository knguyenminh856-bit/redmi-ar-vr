# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Update `ARCHITECTURE.md`, `TASKS.md`, `README.md` | [x] | Removed depth occlusion; documented OTA & GPU hands |
| 2. Rust Core | Update `hand_tracking.rs`, `finger_draw.rs`, `scene_3d.rs`, `renderer.rs`, `lib.rs` (delete `depth_occlusion.rs`) | [x] | 12 unit tests pass |
| 3. Android Shell | Add `debug-fixed.keystore`, `ApkFileProvider.kt`, `AppUpdater.kt`, GPU `480p` MediaPipe, Quick Bar + HUD | [x] | Verified locally |
| 4. CI/CD APK | Update `.github/workflows/build-apk.yml` (`fetch-depth: 0` + `BuildCommit`/`BuildNumber`) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Run `cargo test`, `cargo check --target aarch64-linux-android`, `gradlew :app:assembleDebug`, push `main` | [x] | Verified & pushed |
