# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #18 — Nhớ trọn phòng 360° chống lộn xộn khi quay lại + Quét sàn TSDF phẳng + Lưu trữ 2,048 chunk)

1. **Nhớ phòng 360° cực vững — Quay qua quay lại không bị mất hay chồng lộn xộn (`spatial_mesh.rs`, `MainActivity.kt`):**
   - **Tăng dung lượng bộ nhớ gần 4 lần (`1,200` chunk ≈ 19 MB RAM, VBO `50,000` tam giác) & giữ trọn 360° trong GPU:** bỏ hoàn toàn việc cắt chunk sau lưng camera ra khỏi VBO nên khi xoay người `180°–360°` tại chỗ, toàn bộ phòng đã quét vẫn nằm nguyên tại chỗ tức thì.
   - **Bảo vệ vùng đã quét (`PROTECTED_WEIGHT = 46`) + Chờ ổn định depth `240ms` sau khi quay đầu:** siết ngưỡng chống rung (`0.90 rad/s`), chờ `240ms` sau mỗi cú xoay để bộ lọc depth ARCore ổn định, bỏ 10% rìa ảnh depth méo, và yêu cầu `2` khung hình liên tiếp rõ nét cách `> 9.5cm` mới được gọt bề mặt đã xác nhận — chặn đứng tình trạng 1 khung hình trễ làm nát tường/bàn vừa quét.
2. **Quét được sàn nhà chuẩn thực tế (tự gộp phẳng, không dùng thảm ARCore giả):**
   - Tắt hoàn toàn `PlaneFindingMode` của ARCore (nhẹ CPU/RAM); lưới TSDF quét thẳng sàn nhà nơi bạn hướng camera tới, tự gộp thành tấm phẳng sạch gọn và đỡ vật lý 100% theo lưới đã quét.
3. **Nâng cấp lưu trữ & định vị `.arroom` v2 (`room_persistence.rs`, `RoomMapStorage.kt`):**
   - Nâng giới hạn file `.arroom` lên `2,048` chunk (trước đây >384 chunk bị từ chối nạp!), tăng gấp đôi mốc ảnh định vị lên `16` mốc (`26°`/mốc), tự lấp khe xoay khi nạp lại, dựng đủ toàn bộ lưới ngay khi khớp và tự lưu cache mỗi `20 giây`.
4. **Chỉ vật đã scan mới là thật (`physics_3d.rs`) + Lưới thích ứng `2.5cm` + Nút bấm lớn + Đèn pin `🔦` + OTA (`🔄` ➔ `⬇ #18`).**

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
