# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính

1. **Nhìn xuyên thấu 120Hz chuẩn tỷ lệ 1:1 & Full màn hình:**
   - **Zero-Copy OES + ARCore Per-Eye + Cutout ALWAYS:** Khung camera mở rộng chuẩn từng mắt (`1200×1080`), khóa tỷ lệ 3D khớp 1:1 với thế giới thật ở mọi mức zoom, tràn nốt ruồi căn giữa chuẩn cho kính VR và ưu tiên làm tươi `120Hz` trực tiếp trên Surface.
   - **Tối ưu RAM (~10 MB) + Bảng LUT 2MB:** Khởi tạo sẵn bảng tra YUV→RGBA trong RAM + bộ đệm xoay vòng 4 khung hình (`vision_ram.rs`) và nạp sẵn model AI vào RAM giúp không cấp phát rác (`Bitmap`), giải phóng tối đa GPU `Mali-G57 MC2`.
2. **Nhận diện tay 5 ngón thông minh (`🖐 Tay: T.ĐỘNG / NHANH / TẮT` & `✏ Vẽ`):**
   - MediaPipe `HandLandmarker` dùng **`Delegate.GPU` kèm Kernel Cache** đọc `ByteBuffer` RGBA đã xoay thẳng góc từ Rust. Tự động hạ về 4Hz khi không có tay, đánh thức nhanh ~12Hz khi thấy màu da/chuyển động, và tăng tốc ~24Hz khi thấy tay.
   - **Vẽ không gian ở `0.65m` (`✏ Vẽ: TẮT / 1 NGÓN / 5 NGÓN`):** Nội suy đường cong mượt + dải phát sáng 2 lớp. Bấm **`🗑`** (hoặc combo `✌ ➔ ✊`) để xóa nét vẽ.
3. **8 Cử chỉ & 5 Combo hành động:**
   - **8 cử chỉ:** `✊ Nắm đấm`, `🤏 Chụm ngón`, `👉 Cây súng`, `✌ Chữ V`, `🖐 Xòe tay`, `👍 Thích`, `🤘 Rock`, `☝ Chỉ 1 ngón`.
   - **5 Combo (trong `1.65s`):** `🤏➔✊` (Hút/Đẩy khối 3D), `✊➔🖐` (Sóng năng lượng), `🤏➔👉` (Đổi màu khối 3D), `✌➔✊` (Xóa nét vẽ), `🤘➔🖐` (Định tâm khối 3D).
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
