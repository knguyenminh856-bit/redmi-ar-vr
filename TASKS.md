# TASKS — Redmi Note 14 4G Rust AR/VR Passthrough

| Phase | Task | Status | Notes |
| :--- | :--- | :--- | :--- |
| 1. Setup | Update `ARCHITECTURE.md`, `TASKS.md`, `README.md`, `.gitignore`, `build.gradle.kts` | [x] | Documented full-screen 2.5D voxels & rigid-body physics |
| 2. Rust Core | Add `physics_3d.rs`, upgrade `room_scanner.rs` (6cm PointCloud voxels + 2.5D AABBs), cover-fit 100% fullscreen projection (`renderer.rs`, `lib.rs`) | [x] | 23 unit tests + `aarch64-linux-android` pass |
| 3. Android Shell | Add `acquirePointCloud` (5Hz), `🎲 Thả` button, `KEY_FOV_PCT_V2=100`, 55..105% slider, compact toolbar & telemetry | [x] | Compiles cleanly (`:app:assembleDebug`) |
| 4. CI/CD APK | GitHub Actions `build-apk.yml` (`arm64-v8a` `.so` + signed `.apk`) | [x] | Rolling `latest-apk` release |
| 5. Verify & Git | Verify tests & Gradle build, clean, commit, push `main` | [x] | Triggered Build #7 on `latest-apk` |
| 6. Build #8 3D Realism | 1-Eye Mono default + 2-Eye toggle, live ARCore HitTest reticle + patch synthesis, world-locked 20cm metric grid, anchored floor, 4x substeps + exact table edges + cube upright settling + shadows & ripples + lazy Hand/Object AI | [x] | Published Build #8 (`ee76dc1`) |
| 7. Build #9 Fusion & Tri-Mesh | Add `spatial_mesh.rs` (ARCore true polygons + Coplanar 2D Convex Hull merging, 128-chunk Sparse Occupancy + Free-Space Ray Carving + Greedy Meshing, GLES 3.0 barycentric `🔺 Lưới △` Triangle Mesh mode), fix 1-Eye HUD `LayoutParams`, remove phantom `InstantPlacement` patches, `near=0.02m`, ergonomic bottom-right Drop/Throw dock & compact telemetry | [x] | 30 Rust tests + `:app:assembleDebug` |
| 8. Build #10 Persistence (Planned) | `.arroom` compressed binary room save/load + `.obj` export + 4-DoF Keyframe/Anchor Relocalization | [ ] | Next milestone |
