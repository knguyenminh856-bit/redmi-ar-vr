# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #16 — Lưới thích ứng 2.5cm + Gộp mặt phẳng + Xóa sàn & Cải cách giao diện nút lớn)

1. **Lưới thích ứng theo khoảng cách (`2.5cm` siêu mịn khi lại gần) & Tự sửa lỗi lơ lửng (`spatial_mesh.rs`):**
   - **Càng lại gần đồ vật gồ ghề (`<= 0.95m`), lưới càng bám sát & tam giác càng nhỏ (`2.5cm`, màu vàng cam):** điểm đo ở gần tự động hạ trọng số dữ liệu cũ ở xa để đè sạch sai lệch cũ; ở xa (`> 0.95m`) tự gộp `2×2` (`5cm`, xanh ngọc) giúp cực nhẹ máy.
   - **Tự động xóa lỗi lơ lửng ngay khi quay cam lại:** gọt không gian rỗng sát đến `5.5cm` trước vật thật (xóa vùng mù gọt cũ) + bộ lọc BFS dọn sạch các cụm vụn lơ lửng (`< 7` voxel) không dính vào vật thể.
2. **Gộp mặt phẳng / hình đơn giản thành 1 tấm sạch & Bỏ hoàn toàn lưới sàn:**
   - **Gộp mặt phẳng (Coplanar Greedy Merging):** tường, mặt bàn, cánh tủ, mặt hộp phẳng tự động hợp nhất thành 1 tấm phẳng lớn màu xanh kiến trúc (không bị chia nát hay dính nhầm tường ARCore).
   - **Xóa bỏ hoàn toàn lưới sàn:** không còn trải thảm/lưới lên sàn nhà (chỉ giữ cao độ sàn tàng hình cho vật lý ném/thả bóng).
3. **Cải cách dự án & Giao diện nút bấm lớn dễ thao tác:**
   - Xóa sạch mã nguồn cũ thừa (`scene_3d.rs`, thẻ nhãn nổi, JNI thừa).
   - Phóng to toàn bộ nút điều khiển (`11.5sp–13.5sp`, chiều cao `38–46dp`), đưa **`🧹 Xóa cache`** lên thẳng thanh trên cùng (7 nút thoáng mắt) và gom các mục phụ (`1/2 Mắt`, `AI Tay`, `Vẽ`, `Thông số`, `Che khuất`) vào bảng **`⚙ Cài đặt`** gọn gàng.
4. **Tự lưu cache vào điện thoại (`auto_room_cache.arroom`) + Đèn pin `🔦` + `⚡ Quét mạnh` ⇄ `🔒 Khóa` + OTA (`🔄` ➔ `⬇ #16`).**

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
