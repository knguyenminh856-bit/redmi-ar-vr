# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính (Build #11 — Lưới Surface Nets mượt `4cm` + Che khuất Z-Buffer 3D)

1. **Lưới 3D Mượt ôm sát thực thể tĩnh (`🔺 Lưới △` mặc định — `spatial_mesh.rs`):**
   - **Độ chi tiết `4cm` + Sub-Voxel Surface Nets (`0.15mm`) + Làm mượt Laplacian:** Nâng độ phân giải khối gấp `3.375×` (192 Sparse Chunks `~3.8 MB` RAM), giữ lại tọa độ bề mặt dưới-centimet của từng điểm 3D và nối thành **lớp lưới tam giác mượt ôm sát vật thể thật** (bỏ hoàn toàn dạng khối hộp vuông cứng).
   - **Tăng cường chi tiết đa nguồn (`RawDepth 3Hz` + `SLAM PointCloud 5Hz` + Đa giác ARCore thật):** Nếu máy hỗ trợ `RAW_DEPTH_ONLY`, app tự lấy thêm lưới độ sâu tin cậy cao (`conf >= 175`, `24×18` điểm @ `3Hz`) hợp nhất qua tối thiểu 3 khung hình — chỉ **thực thể tĩnh** mới khóa thành lưới chắc chắn, còn đồ đã dời bị **Free-Space Carving (`3.6cm`)** tự gọt sạch.
2. **Che khuất tự động bằng Hardware Z-Buffer (`🛡 Che khuất vật thật: BẬT`):**
   - Dùng chính mô hình lưới 3D tĩnh đã hợp nhất để ghi độ sâu phần cứng (`color_mask = false`, đẩy lùi `1.6cm` theo pháp tuyến để vật đặt *trên* bàn không bị cắt):
     - **Tự che các lớp lưới scan bị khuất phía sau** tủ, ghế, giường, tường (không còn rối mắt khi nhìn xuyên qua vật thể).
     - **Tự che khuất bóng/khối vật lý và nét vẽ 3D** khi lăn hoặc nằm ra sau đồ vật thật — ổn định 120Hz, không nhấp nháy như depth từng khung hình. (Có nút bật/tắt `🛡 Che khuất` trong menu `⚙`).
3. **Lưu / Nạp Phòng 3D Offline (`💾 Phòng`), Xuất `.obj` & Tự khớp 4-DoF:**
   - Lưu/nạp bản đồ phòng siêu gọn `.arroom` trên luồng I/O nền (`Android/data/com.minhkhoi.arvr/files/rooms/`), xuất `.obj` mở được trên Blender/PC, và tự khớp lại căn phòng bằng 8 mốc ảnh `32×24` (Shifted-NCC) kèm nút **`📍 Khớp ngay`**.
4. **Chế độ `👁 1 Mắt` / `🥽 2 Mắt`, Vật lý 4x Substeps & Tự cập nhật OTA (`🔄` ➔ `⬇ #11`):**
   - Mặc định **`👁 1 Mắt`** Full màn `2400×1080`, nút **`⬇ Thả`** & **`🎯 Ném`** góc dưới phải; bấm **`🔄`** để cập nhật thẳng APK mới nhất từ GitHub Releases (`latest-apk`).

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
