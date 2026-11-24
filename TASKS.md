# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Create `ARCHITECTURE.md`, `TASKS.md`, `.gitignore`, `README.md` | [x] | Done |
| 2. Rust Core | Implement `Cargo.toml`, `gyro_predict.rs`, `passthrough.rs`, `scene_3d.rs`, `hand_tracking.rs`, `renderer.rs`, `lib.rs` | [x] | 10 unit tests pass |
| 3. Android Shell | Create Gradle DSL project, `AndroidManifest.xml`, `NativeBridge.kt`, `HandTrackerHelper.kt`, `MainActivity.kt` | [x] | Compiles with AGP 8.7.3, ARCore 1.45.0 & MediaPipe 1.2.3 |
| 4. CI/CD APK | Create `.github/workflows/build-apk.yml` (Rust `cargo-ndk` + Gradle APK + Release) | [x] | Rolling `latest-apk` release + artifact |
| 5. Verify & Git | Run `cargo test`, `cargo check --target aarch64-linux-android`, `gradlew :app:assembleDebug`, push `main` | [x] | Verified & pushed to GitHub |
