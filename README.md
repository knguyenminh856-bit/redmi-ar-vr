# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #13 — Chống bóng ma không khí 4 lớp + Khóa lưới thật `🔒 Khóa`)

1. **Chống vẽ nhầm trên không khí (4 lớp lọc phần cứng & hình học — `spatial_mesh.rs`, `MainActivity.kt`):**
   - **Lọc độ tin cậy nghiêm ngặt (`acquireRawDepthConfidenceImage`):** Loại bỏ hoàn toàn pixel nội suy ảo giữa không trung (`conf < 65` khi Quét mạnh, `conf < 120` khi Khóa), giới hạn tầm chính xác `0.30m – 2.85m`.
   - **Chống rung mờ camera (*Motion-Blur Gate*) + chống trùng khung hình (`depthImg.timestamp`):** Tự động tạm bỏ qua các frame depth khi tay quay nhanh (`> 65°/s` hoặc `> 1.1 m/s`) và không bao giờ cộng dồn cùng 1 frame depth cũ nhiều lần.
   - **Lọc biên 2 tầng (`±1px` & `±2px`) + khóa đa góc nhìn (`MESH_MIN_WEIGHT = 42`, 3–5 frame độc lập) + tỉa bụi lơ lửng 6 láng giềng:** Chỉ các thực thể tĩnh thật mới bám lưới tam giác mượt `3.5cm` (bỏ hẳn đốm rác giữa không khí).
2. **Nút 1 chạm `⚡ Quét mạnh` (9Hz liên tục) ⇄ `🔒 Khóa` (2Hz Khóa cứng — chỉ đổi khi thật sự thay đổi):**
   - **`⚡ Quét mạnh` (màu cam):** Quét sâu `9Hz` liên tục giúp lưới TSDF ôm kín cốc, laptop, bàn ghế, góc tường.
   - **`🔒 Khóa` (màu xanh teal — tự bật sau khi nạp phòng `.arroom`):**
     - **Khóa thật 100%:** quay sang góc chưa quét cũng **0 cập nhật**, dao động độ sâu tự nhiên `±11cm` của camera đơn cũng **0 cập nhật** (`🔒 Đã khóa`), không rung, không quét lan sang không khí.
     - **Xác nhận đổi 3 nhịp liên tiếp (`~1.5s`, bộ đếm bit-packed không trùng tia):** chỉ khi có vật thật bị nhấc đi hoặc vật mới đặt tĩnh ngay trước vùng đã khóa suốt `≥ 3` nhịp (`1.5s`), app mới tự gọt hoặc bù đúng chỗ đó.
     - Kèm nút **`🧹 Xóa lưới`** trong menu **`💾 Phòng`** để xóa sạch lưới cũ và quét lại trong 1 chạm.
3. **Che khuất vật bị khuất bằng Hardware Z-Buffer (`🛡 Che khuất vật thật: BẬT`) & `.arroom` v2 (TSDF):**
   - Tự che khuất lớp lưới phía sau đồ vật lẫn bóng/khối vật lý 3D khi lăn ra sau bàn/ghế/cốc ở 120Hz; lưu/nạp phòng TSDF `.arroom` v2, xuất `.obj`, tự cập nhật OTA (`🔄` ➔ `⬇ #13`).

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
