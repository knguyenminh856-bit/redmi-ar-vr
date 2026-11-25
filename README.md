# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #15 — Sửa dính tường giả + Bỏ AI quét vật lỗi + Tự lưu cache vào máy & Xóa cache)

1. **Sửa triệt để lỗi vật thể bị dính nhãn "Tường" vĩnh viễn (`spatial_mesh.rs`):**
   - **Ngưỡng Tường kiến trúc chuẩn phòng thật:** chỉ bề mặt phẳng đứng kích thước thật (`>= 1.15 m²` và `>= 1.25m × 0.72m`) mới được coi là Tường phẳng tầng 1 — màn hình, tủ nhỏ, lưng ghế, thùng hộp không bao giờ bị ép thành tường xanh.
   - **Tự động gỡ / hạ cấp Tường giả theo thời gian thực:** khi tia độ sâu xuyên qua sau tấm tường cũ (`> 16cm`, ví dụ mở cửa hoặc dời đồ) thì **tự động xóa tấm tường đó**; khi thấy bề mặt gồ ghề nhiều chi tiết (`curv >= 18mm`) thì **tự hạ cấp về vật thể 3D** để lưới chi tiết `3.5cm` TSDF ôm sát trở lại.
2. **Tự động lưu bản đồ 3D vào bộ nhớ điện thoại & tự nạp lần sau (`RoomMapStorage.kt`):**
   - **Tự lưu (`filesDir/rooms/auto_room_cache.arroom`):** tự động lưu nguyên bản đồ TSDF + 8 mốc ảnh định vị ngay khi bạn bấm **`🔒 Khóa`**, khi thoát/ẩn app (`onPause`), và định kỳ mỗi 35 giây.
   - **Tự nạp khi mở lại app:** tự bật thanh khớp phòng 4-DoF ở góc dưới trái (**`📍 Khớp ngay`** hoặc lia nhẹ camera để tự khớp).
   - **Nút 1 chạm `🧹 Xóa cache cũ`** ở cả 3 vị trí: ngay thanh khớp phòng góc dưới trái, trong mục **`💾 Phòng`** (hiện dung lượng cache đang lưu), và trong cài đặt **`⚙`**.
3. **Gỡ bỏ hoàn toàn tính năng lỗi AI quét vật (`RoomScannerHelper.kt`, `efficientdet_lite0.tflite`):**
   - Giảm ~4.5 MB dung lượng APK, thanh chế độ quét gọn sạch chỉ còn 3 nấc: **`🔺 Lưới △`** ➔ **`👁 Ẩn lưới`** ➔ **`⏹ TẮT`**.
4. **Đèn pin `🔦` + `⚡ Quét mạnh` (9Hz) ⇄ `🔒 Khóa` (2Hz) + Che khuất Z-Buffer + OTA (`🔄` ➔ `⬇ #15`):**
   - Bật đèn LED sau ngay trong ARCore khi quét góc tối; che khuất vật lý 3D sau đồ thật 120Hz.

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
