# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính (Build #10 — Lưu/Nạp Phòng 3D `.arroom`, Xuất `.obj` & Tự khớp 4-DoF)

1. **Lưu / Nạp Bản đồ Phòng 3D Offline (`💾 Phòng`) & Tự khớp 4-DoF:**
   - **Định dạng nhị phân siêu gọn `.arroom` (`room_persistence.rs` & `RoomMapStorage.kt`):** Lưu mặt phẳng đã gộp, các ô khối 3D thưa (3 byte/ô) và tối đa `8` mốc chữ ký ảnh `32×24` hoàn toàn trên luồng I/O nền (không khựng 120Hz) vào thư mục `Android/data/com.minhkhoi.arvr/files/rooms/`.
   - **Tự khớp lại phòng không cần mạng (4-DoF Relocalization):** Khi mở app lần sau và bấm **`📂 Nạp`**, app tự đối chiếu ảnh camera với `8` mốc ảnh bằng thuật toán NCC chống chói sáng (`4Hz`). Khớp xong tự căn góc quay `yaw` + dịch chuyển `(tx, ty, tz)` rồi **gộp thẳng bản đồ cũ vào phiên mới** để tiếp tục quét tăng cường & gọt bỏ đồ đã di chuyển! Có sẵn nút **`📍 Khớp ngay`** để khóa thủ công tức thì khi ánh sáng phòng đổi mạnh.
   - **Xuất mô hình 3D chuẩn (`📤 Xuất .OBJ`):** Xuất toàn bộ lưới tam giác căn phòng (`Floor`, `Table`, `Seat`, `Wall`, `Obstacles`) ra file `.obj` mở được trực tiếp trên Blender / PC.
2. **Chế độ Lưới Tam Giác (`🔺 Lưới △`) & Tái tạo 3D không chồng lớp (`spatial_mesh.rs`):**
   - Bấm nút góc trên phải chuyển qua **`🔺 Lưới △`** để nhìn chính xác từng **tấm lưới tam giác** (đa giác mặt phẳng thật đã gộp đồng phẳng + bề mặt vật cản 3D từ **Greedy Meshing** trên 128 Sparse Chunks `6cm`), vẽ bằng shader GLES 3.0 `fwidth` siêu nét trong 1 draw call.
   - **Tự xóa vật cũ khi dời đồ (Free-Space Carving):** Chiếu tia bước `5.5cm` xuyên qua ô cũ nên di chuyển bàn ghế xong mô hình 3D tự cập nhật không bóng ma; `InstantPlacement` chỉ hiện vòng nét đứt ước lượng.
   - Chu kỳ nút quét: **`🏠 Lưới 3D` ➔ `🔺 Lưới △` ➔ `🏠 Ẩn lưới` ➔ `🏠 +AI Vật` ➔ `🏠 TẮT`**.
3. **Chế độ `👁 1 Mắt` / `🥽 2 Mắt`, Soi cận cảnh (`2cm`) & Nút Vật lý góc dưới phải:**
   - Mặc định **`👁 1 Mắt`** Full màn hình `2400×1080` chuẩn khoảng cách; nút **`⬇ Thả`** và **`🎯 Ném`** to dễ bấm ở góc dưới phải; vật lý 4x substeps, bóng đổ 2 lớp và vòng sóng va chạm.
4. **Bảng thông số gọn (`📊`) & Tự động cập nhật OTA (`🔄` ➔ `⬇ #10`):**
   - Hiển thị gọn Render FPS/ms, số mặt phẳng gộp, chunk 3D, số tam giác `△`, số mốc ảnh định vị `X/8`, Pin/Nhiệt độ/RAM; bấm **`🔄`** để tải và cài đè bản mới nhất trực tiếp từ GitHub Releases (`latest-apk`).

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
