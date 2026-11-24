# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính

1. **Nhìn xuyên thấu Full Màn Hình 120Hz & Khóa khớp 1:1:**
   - **Full-Screen Cover-Fit + Cutout ALWAYS:** Khung camera tự phủ kín 100% khung hình hai mắt (`1200×1080`) không còn viền đen xung quanh (kèm 6% overscan chống hở mép), trong khi vẫn giữ tỷ lệ 3D khớp 1:1 với vật thật.
   - **Tối ưu RAM + Bảng LUT 2MB:** Khởi tạo sẵn bảng tra YUV→RGBA trong RAM + bộ đệm xoay vòng 4 khung hình (`vision_ram.rs`) và nạp sẵn model AI vào RAM giúp không cấp phát rác (`Bitmap`), giải phóng tối đa GPU `Mali-G57 MC2`.
2. **Nhận diện tay 5 ngón thông minh (`🖐 T.Đ / NHANH / TẮT` & `✏ Vẽ`):**
   - MediaPipe `HandLandmarker` dùng **`Delegate.GPU` kèm Kernel Cache** đọc `ByteBuffer` RGBA đã xoay thẳng góc từ Rust. Tự động hạ về 4Hz khi không có tay, đánh thức nhanh ~12Hz khi thấy màu da/chuyển động, và tăng tốc ~24Hz khi thấy tay.
   - **Cử chỉ 👉 Cây súng & Đánh tay:** Giơ tay hình **👉 Cây súng** để bắn bóng nảy vật lý; vung ngón tay hất/đẩy các khối đang nằm trong phòng!
3. **Bản đồ 2.5D SLAM + Vật lý 120Hz (`🎲 Thả` & `🏠 2.5D`):**
   - **Tái tạo khối 2.5D không cần Depth GPU:** Kết hợp mặt phẳng ARCore + đám mây điểm `PointCloud` 5Hz thành khối Voxel `6cm` + hộp đồ vật `EfficientDet` (`room_scanner.rs`).
   - **Vật lý thực (`physics_3d.rs`):** Bấm **`🎲 Thả`** để ném khối lập phương / bóng có trọng lực (`9.81 m/s²`), bóng đổ tiếp xúc dưới sàn, **tự rơi đậu lên mặt bàn/sàn** và **bị tường / vật thể 2.5D cản lại hoặc bật nảy**. Bấm **`🗑`** để xóa cả nét vẽ lẫn vật thể vật lý.
4. **Tự động cập nhật trong app (`🔄 Check` ➔ `⬇ Cập nhật`):**
   - App tự kiểm tra bản build mới nhất trên GitHub Releases (`latest-apk`), hiển thị nút xanh **`⬇ Cập nhật`** khi có bản mới, tải trực tiếp với % tiến độ và mở trình cài đặt đè ngay không mất dữ liệu.
5. **Bảng thông số chi tiết của máy (`📊 Info`):**
   - Hiển thị đồng bộ 2 mắt rõ nét (không bị thanh nút che đè): phiên bản build, tên máy/SoC, GPU OpenGL, màn hình ngang, Render FPS/ms, AI Tay FPS/ms & thời gian YUV LUT (`µs`), số mặt phẳng (kèm bộ nhớ RAM), số đồ vật, tọa độ ngón trỏ `(X, Y, Z)`, Pin `%`, Nhiệt độ `°C`, RAM `MB`.
6. **Nhận diện Sàn, Tường, Mặt bàn & Đồ vật nhớ bằng RAM (`🏠 ĐẦY ĐỦ / SÀN/TƯỜNG / TẮT`):**
   - **Bộ nhớ không gian RAM (`45s` mặt phẳng / `10s` đồ vật 3D):** Lọc chuẩn kích thước **Tường** thật (loại các mặt nhỏ như vỏ PC thành **Mặt đứng**), căn thẳng theo trọng lực, tự thu nhỏ theo khoảng cách (`scissor` gọn trong khung camera) và ghim nhãn đồ vật ngay trong không gian 3D.

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí 2 khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
