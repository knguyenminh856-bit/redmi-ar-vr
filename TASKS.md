# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Create `ARCHITECTURE.md`, `TASKS.md`, `.gitignore`, `README.md` | [x] | Updated with Depth, Combos & Telemetry |
| 2. Rust Core | `depth_occlusion.rs`, `finger_draw.rs`, `hand_tracking.rs` (1€ + 8 poses + 5 combos), `scene_3d.rs`, `renderer.rs`, `lib.rs` | [x] | 14 unit tests pass |
| 3. Android Shell | `FastEglConfigChooser`, ARCore 480p/1080p split + `DEPTH16`, CPU MediaPipe `224px`, Quick Bar, Stereo Telemetry HUD | [x] | Verified locally |
| 4. CI/CD APK | `.github/workflows/build-apk.yml` (Rust `cargo-ndk` + Gradle APK + Release) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Run `cargo test`, `cargo check --target aarch64-linux-android`, `gradlew :app:assembleDebug`, push `main` | [x] | Pushed & verified |
