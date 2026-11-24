# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính (Build #8 — Hoàn thiện Không gian 3D & Vật lý thực)

1. **Chế độ 1 Mắt / 2 Mắt & Toàn màn hình 120Hz (`👁 1 Mắt` / `🥽 2 Mắt`):**
   - **Mặc định `👁 1 Mắt` AR toàn màn (`2400×1080`, zero-IPD):** Loại bỏ hiện tượng lệch thị sai 64mm khi đưa điện thoại sát bàn/đồ vật để đo khoảng cách & test vật lý chính xác tuyệt đối; bấm 1 chạm chuyển sang **`🥽 2 Mắt`** cho hộp kính VR Stereo.
   - **Tạm tắt AI Tay mặc định (`🖐 TẮT`):** Chỉ nạp TFLite/GPU khi người dùng chủ động bật, giữ Render mượt ~120fps.
2. **Hồng tâm ARCore 3D & Lưới kích thước thực (`🎯 Mặt bàn · 0.64m` & `🏠 Lưới 3D`):**
   - Bắn tia `frame.hitTest` / `InstantPlacement (LOCAL_Y_UP)` theo thời gian thực để vẽ vòng hồng tâm ôm sát độ nghiêng mặt phẳng thực, hiển thị khoảng cách mét ở cạnh dưới và tự tổng hợp mảnh mặt phẳng `36cm` tức thì tại điểm chạm.
   - **Lưới 20cm khóa cứng thế giới:** Phủ lưới góc phối cảnh mảnh (`20cm/ô`) lên sàn, mặt bàn, ghế và tường; hỗ trợ 4 chế độ (`Lưới 3D` ➔ `Ẩn lưới` ➔ `+AI Vật` ➔ `TẮT`).
3. **Vật lý thực tế 4x Substeps (`⬇ Thả`, `🎯 Ném`, `🗑`):**
   - **Khóa sàn thế giới thực (`anchored_floor_y`):** Không bị tụt sàn hay trôi xa ra khi cúi người/đưa camera lại gần vật thể.
   - **`⬇ Thả` & `🎯 Ném`:** `⬇ Thả` thả rơi tự do vuông góc ngay phía trên điểm hồng tâm; `🎯 Ném` phóng vật theo cung parabol bù trọng lực tới đúng điểm đích đang nhắm.
   - **Hành vi thực:** Khối lập phương tự xoay cân bằng mặt đáy nằm phẳng khi đáp lên bàn, phanh ma sát tĩnh chống trượt vô tận, lăn qua mép bàn là rơi xuống sàn bên dưới, bóng đổ 2 lớp (`umbra + penumbra`) và vòng sóng gợn khi va chạm mạnh.
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
