# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Update `ARCHITECTURE.md`, `TASKS.md`, `README.md`, `.gitignore`, `build.gradle.kts` | [x] | Documented room scanning + wired `efficientdet_lite0.tflite` |
| 2. Rust Core | Add `room_scanner.rs` + wire `renderer.rs` & `lib.rs` | [x] | 15 unit tests + `aarch64-linux-android` check pass |
| 3. Android Shell | Add `RoomScannerHelper.kt` (CPU `ObjectDetector`), YUV rewind, ARCore Planes, Stereo Mini-Tags, `🏠 Quét` button | [x] | Compiles cleanly with `:app:assembleDebug` |
| 4. CI/CD APK | GitHub Actions `build-apk.yml` (`arm64-v8a` `.so` + signed `.apk`) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Verify tests & Gradle build, clean, commit, push `main` | [x] | Triggered Build #5 on `latest-apk` |
