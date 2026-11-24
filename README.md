# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #12 — Lưới TSDF ôm sát vật thể `3.5cm` + Quét Mạnh / Duy Trì)

1. **Lưới TSDF ôm khít thực thể thật (Bỏ hoàn toàn mặt phẳng xếp chồng — `spatial_mesh.rs`):**
   - **ARCore Dense Depth (`DepthMode.AUTOMATIC`, zero-copy `ByteBuffer`):** Dùng chính `acquireDepthImage16Bits()` kết hợp `cam.imageIntrinsics` + `cam.pose` chuẩn cảm biến và lọc nhảy cạnh bóng ma (*silhouette jump guard*) để chiếu chính xác từng điểm bề mặt của cốc, laptop, bàn phím, ghế, tủ, tường vào không gian 3D.
   - **Sparse TSDF `3.5cm` + Surface Nets cắt điểm 0 (`256` chunks, tối đa `30.000△` ưu tiên vật gần):** Nội suy tuyến tính chính xác mặt `tsdf = 0` cho độ sắc nét dưới-centimet, tự phân màu theo pháp tuyến (sàn xanh ngọc, mặt bàn/đỉnh vật vàng hổ phách, thành đứng/vật cong xanh lá-lam), bật `CULL_FACE(BACK)` một mặt duy nhất, **không còn vẽ đè các tấm mặt phẳng ARCore lên vật thể**.
2. **Nút 1 chạm `⚡ Quét mạnh` (10Hz liên tục) ⇄ `🛡 Duy trì` (2.5Hz so sánh & giữ nguyên):**
   - **`⚡ Quét mạnh` (BẬT):** Quét độ sâu liên tục `~10Hz` (độ phân giải đầy đủ `1×`, dựng tới `6` chunk/frame) giúp lưới 3D bám kín nhanh từng đồ vật ngay khi lia máy.
   - **`🛡 Duy trì` (TẮT quét mạnh / Tự bật sau khi nạp phòng `.arroom`):** Hạ tốc độ xuống `2.5Hz` (`stride = 2`), **so sánh chỉ-đọc với TSDF đang khóa trước** — vùng không đổi giữ nguyên `0` cập nhật (nhẹ CPU/GPU, không rung), chỉ quét bổ sung hoặc gọt sạch đúng khu vực có đồ vật mới đặt vào hoặc vừa dời đi.
3. **Che khuất vật bị khuất bằng Hardware Z-Buffer (`🛡 Che khuất vật thật: BẬT`):**
   - Pass 0 ghi chính lưới TSDF ôm sát vật thể vào Depth Buffer (`-1.4cm` bias theo pháp tuyến): tự động **che lớp lưới phía sau đồ vật** và **che bóng/khối vật lý, nét vẽ 3D** khi lăn ra sau bàn/ghế/cốc/tường ở 120Hz.
4. **Lưu / Nạp Phòng 3D `.arroom` v2 (TSDF), Xuất `.obj` & Tự cập nhật OTA (`🔄` ➔ `⬇ #12`):**
   - Lưu/nạp nguyên trường TSDF gọn nhẹ trên luồng nền (`Android/data/com.minhkhoi.arvr/files/rooms/`), tương thích ngược file v1 cũ, tự khớp 4-DoF và tự chuyển sang **`🛡 Duy trì`** sau khi khớp phòng.

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
