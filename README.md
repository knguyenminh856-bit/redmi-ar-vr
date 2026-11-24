# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore + MediaPipe GPU**.

## Tính năng chính (Build #9 — Tái tạo 3D Tăng cường & Chế độ Lưới Tam Giác)

1. **Chế độ Lưới Tam Giác (`🔺 Lưới △`) & Tái tạo 3D không chồng lớp (`spatial_mesh.rs`):**
   - Bấm nút góc trên phải chuyển qua **`🔺 Lưới △`** để nhìn chính xác từng **tấm lưới tam giác** (cả đa giác mặt phẳng ARCore thật đã hợp nhất đồng phẳng lẫn bề mặt vật cản 3D từ thuật toán **Greedy Meshing** trên 128 Sparse Chunks `6cm`), vẽ bằng shader GLES 3.0 `fwidth` siêu nét trong 1 draw call.
   - **Tự xóa vật cũ khi dời đồ (Free-Space Carving):** Chiếu tia tẩy khoảng trống xuyên qua ô cũ nên di chuyển bàn ghế xong mô hình 3D tự cập nhật không bị bóng ma chồng chéo; bỏ hoàn toàn ô vuông `InstantPlacement` rác (chỉ hiện vòng nét đứt ước lượng).
   - Chu kỳ chế độ nút quét: **`🏠 Lưới 3D` (mặt sạch) ➔ `🔺 Lưới △` (xem lưới tam giác) ➔ `🏠 Ẩn lưới` ➔ `🏠 +AI Vật` ➔ `🏠 TẮT`**.
2. **Chế độ `👁 1 Mắt` / `🥽 2 Mắt` Full 120Hz & Soi cận cảnh (`near = 0.02m`):**
   - Đã sửa hoàn toàn lỗi ẩn HUD ở chế độ **`👁 1 Mắt`**: hiện đầy đủ nhãn khoảng cách `🎯 Mặt bàn · 0.64m`, số lượng tam giác `🔺 X△`, banner thông báo ở đáy và bảng `📊` thu gọn không đè tầm nhìn.
   - Giảm mặt cắt gần xuống `2cm` giúp đưa camera dí sát khối 3D không bị cắt thủng.
3. **Nút thao tác vật lý dễ bấm (`⬇ Thả`, `🎯 Ném`) ở góc dưới phải:**
   - Tách riêng 2 nút **`⬇ Thả`** (rơi vuông góc xuống điểm hồng tâm) và **`🎯 Ném`** (cung parabol bù trọng lực) kích thước to ở góc dưới phải vừa tầm ngón cái; giữ vật lý 4x substeps, tự đứng phẳng mặt đáy khối lập phương, bóng đổ 2 lớp và vòng sóng va chạm.
4. **Tự động cập nhật trong app (`🔄` ➔ `⬇ #9`):**
   - Bấm **`🔄`** rồi **`⬇ #9`** ngay trong app để tải và cài đè bản mới nhất trực tiếp từ GitHub Releases (`latest-apk`).
5. **Bảng thông số chi tiết của máy (`📊 Info`):**
   - Hiển thị đồng bộ 2 mắt rõ nét (không bị thanh nút che đè): phiên bản build, tên máy/SoC, GPU OpenGL, màn hình ngang, Render FPS/ms, AI Tay FPS/ms & thời gian YUV LUT (`µs`), số mặt phẳng (kèm bộ nhớ RAM), số đồ vật, tọa độ ngón trỏ `(X, Y, Z)`, Pin `%`, Nhiệt độ `°C`, RAM `MB`.
6. **Nhận diện Sàn, Tường, Mặt bàn & Đồ vật nhớ bằng RAM (`🏠 ĐẦY ĐỦ / SÀN/TƯỜNG / TẮT`):**
   - **Bộ nhớ không gian RAM (`45s` mặt phẳng / `10s` đồ vật 3D):** Lọc chuẩn kích thước **Tường** thật (loại các mặt nhỏ như vỏ PC thành **Mặt đứng**), căn thẳng theo trọng lực, tự thu nhỏ theo khoảng cách (`scissor` gọn trong khung camera) và ghim nhãn đồ vật ngay trong không gian 3D.

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí 2 khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
