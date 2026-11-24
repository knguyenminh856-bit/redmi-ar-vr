# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính

1. **Nhìn xuyên thấu 120Hz chống chóng mặt:**
   - **Zero-Copy OES Camera:** Luồng camera phần cứng đưa thẳng vào GPU ở độ phân giải sắc nét (`≥1080p` GPU, `≤480p` cho AI), tắt Depth API & MSAA 4x để app luôn nhẹ và mát máy.
   - **1:1 Angular Scale + Async Timewarp:** Cửa sổ camera khớp tỷ lệ thật ở `2.5m` trên nền đen AMOLED kèm bù xoay đầu `TYPE_GAME_ROTATION_VECTOR` 120Hz.
2. **Nhận diện tay 5 ngón mượt mà & Vẽ neon 3D (`✏ Vẽ`):**
   - MediaPipe `HandLandmarker` chạy **GPU Delegate** trên ảnh `480px` với bộ lọc làm mượt theo vận tốc, vẽ 5 đường xương nối 21 khớp + tâm ngắm 5 đầu ngón tay.
   - **Vẽ không gian ở `0.65m` (`✏ Vẽ: TẮT / 1 NGÓN / 5 NGÓN`):** Nội suy đường cong mượt + dải phát sáng 2 lớp. Bấm **`🗑 Xóa`** (hoặc combo `✌ ➔ ✊`) để xóa nét vẽ.
3. **8 Cử chỉ & 5 Combo hành động:**
   - **8 cử chỉ:** `✊ Nắm đấm`, `🤏 Chụm ngón`, `👉 Cây súng`, `✌ Chữ V`, `🖐 Xòe tay`, `👍 Thích`, `🤘 Rock`, `☝ Chỉ 1 ngón`.
   - **5 Combo (trong `1.65s`):** `🤏➔✊` (Hút/Đẩy khối 3D), `✊➔🖐` (Sóng năng lượng), `🤏➔👉` (Đổi màu khối 3D), `✌➔✊` (Xóa nét vẽ), `🤘➔🖐` (Định tâm khối 3D).
4. **Tự động cập nhật trong app (`🔄 Check cập nhật` ➔ `⬇ Cập nhật`):**
   - App tự kiểm tra bản build mới nhất trên GitHub Releases (`latest-apk`), hiển thị nút xanh **`⬇ Cập nhật`** khi có bản mới, tải trực tiếp với % tiến độ và mở trình cài đặt đè ngay không mất dữ liệu.
5. **Bảng thông số chi tiết của máy (`📊 Thông số`):**
   - Hiển thị đồng bộ 2 mắt: phiên bản build, tên máy/SoC, GPU OpenGL, tần số màn hình, Render FPS/ms, AI Tay FPS/ms, số mặt phẳng & đồ vật đang quét, tọa độ đầu ngón trỏ `(X, Y, Z)`, Pin `%`, Nhiệt độ `°C`, RAM `MB`.
6. **Nhận diện Sàn, Tường, Mặt bàn & Đồ vật (`🏠 Quét: ĐẦY ĐỦ / SÀN/TƯỜNG / TẮT`):**
   - **ARCore Spatial Planes (4Hz):** Tự phân loại **Sàn**, **Tường**, **Mặt bàn**, **Bề mặt ghế**, **Trần nhà** và vẽ các mảng hình học mờ nhỏ (`alpha ~ 0.18`) bám đúng mặt phẳng thật.
   - **Nhận diện đồ vật CPU (`EfficientDet-Lite0` INT8 @ ~2.6Hz):** Nhận diện bàn, ghế, sofa, giường, màn hình/TV, chai/cốc, cây xanh... kèm khung góc mờ nhỏ + nhãn nổi 2 mắt nhỏ gọn, không làm tụt FPS tay hay xuyên thấu.

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí 2 khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
