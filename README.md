# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #17 — Chỉ vật đã scan mới là thật: Xóa sạch mặt phẳng tự tạo cho Item)

1. **Xóa sạch 100% mặt phẳng tự tạo / đoán ảo (`physics_3d.rs`, `spatial_mesh.rs`, `MainActivity.kt`):**
   - **Không còn miếng đỡ tự dựng (`ensure_explicit_drop_patch`), không còn sàn ảo `fallback_floor_y`, không còn `InstantPlacement` hay tấm `Plane` ARCore tàng hình.**
   - **Hồng tâm 120Hz & Vật lý va chạm 100% theo lưới TSDF đã quét (`raycast_scanned_surface`, `sample_scanned_collision`):** chỉ những bề mặt đã thực sự quét ra lưới mới bắt hồng tâm, đỡ/bật bóng-khối và nhận bóng đổ. Nếu thả/ném item vào vùng chưa quét, item sẽ rơi tự do xuyên qua khoảng không (không có sàn ảo đỡ và không có bóng giả lơ lửng).
2. **Lưới thích ứng theo khoảng cách (`2.5cm` siêu mịn khi lại gần) & Tự sửa lỗi lơ lửng (`spatial_mesh.rs`):**
   - **Càng lại gần đồ vật gồ ghề (`<= 0.95m`), lưới càng bám sát & tam giác càng nhỏ (`2.5cm`, màu vàng cam):** điểm đo ở gần tự động hạ trọng số dữ liệu cũ ở xa để đè sạch sai lệch cũ; ở xa (`> 0.95m`) tự gộp `2×2` (`5cm`, xanh ngọc) giúp cực nhẹ máy.
   - **Tự động xóa lỗi lơ lửng ngay khi quay cam lại:** gọt không gian rỗng sát đến `5.5cm` trước vật thật (xóa vùng mù gọt cũ) + bộ lọc BFS dọn sạch các cụm vụn lơ lửng (`< 7` voxel) không dính vào vật thể.
3. **Gộp mặt phẳng / hình đơn giản thành 1 tấm sạch & Bỏ hoàn toàn lưới sàn:**
   - **Gộp mặt phẳng (Coplanar Greedy Merging):** tường, mặt bàn, cánh tủ, mặt hộp phẳng tự động hợp nhất thành 1 tấm phẳng lớn màu xanh kiến trúc.
   - **Xóa bỏ hoàn toàn lưới sàn:** không còn trải thảm/lưới lên sàn nhà và không tạo sàn va chạm ảo.
4. **Giao diện nút bấm lớn (`11.5sp–13.5sp`, cao `38–46dp`) + Tự lưu cache (`auto_room_cache.arroom`) + Đèn pin `🔦` + `⚡ Quét mạnh` ⇄ `🔒 Khóa` + OTA (`🔄` ➔ `⬇ #17`).**

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
