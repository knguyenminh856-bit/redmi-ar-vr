# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth API + MediaPipe GPU**.

## Tính năng chính (Build #14 — Tách riêng Vùng Chi Tiết vs Mặt Phẳng/Tường + Nút Đèn Pin `🔦`)

1. **Tách biệt rõ 2 tầng không gian 3D (`spatial_mesh.rs`):**
   - **Tầng 1 — Mặt phẳng cấu trúc (Sàn / Tường / Trần):** Lưới kiến trúc khổ lớn (`~38cm`) phẳng sạch, không gồ ghề: **Sàn xanh ngọc dịu**, **Tường xanh dương**, đẩy lùi nhẹ `-6mm` để không bao giờ tranh chấp chiều sâu với đồ vật đặt trên sàn hay dựa tường.
   - **Tầng 2 — Vùng nhiều chi tiết & Đồ vật 3D (`3.5cm` Sparse TSDF Surface Nets):** Mọi vật thể nhô khỏi tường/sàn hoặc có độ cong/cạnh phức tạp (bàn, ghế, laptop, cốc, chai nước, bàn phím, hộp) được bọc bằng lưới tam giác dày đặc `3.5cm`:
     - **Vùng cạnh cong / cực nhiều chi tiết:** nổi bật màu **vàng hổ phách** theo độ cong 3D thực tế.
     - **Bề mặt đồ vật nhẵn / mặt bàn:** màu **xanh ngọc lục bảo** tách bạch hoàn toàn với tường và sàn.
   - **Khắc phục triệt để lỗi mất lưới ở Build #13 mà vẫn chặn bóng ma không khí:** dùng lọc độ tin cậy neo vùng lân cận (`±4px`, cho phép mặt nhẵn của vật thật dính liền vào cạnh rõ nét, vẫn chặn khoảng không `conf=0` lơ lửng), giữ chống rung mờ camera (`1.5 rad/s`), chống trùng `timestamp` và `MESH_MIN_WEIGHT = 28`.
2. **Nút bật đèn pin `🔦` ngay trên thanh công cụ (`Config.FlashMode.TORCH` + Camera2 fallback):**
   - Chạm **`🔦` ➔ `🔦 BẬT`** để bật đèn LED sau ngay trong phiên ARCore — chiếu sáng rõ góc tối, tăng tốc độ màn trập giảm mờ nhòe và giúp quét chi tiết đồ vật sắc nét gấp nhiều lần (tự tắt khi thoát app để chống nóng máy).
3. **Chế độ `⚡ Quét mạnh` (9Hz) ⇄ `🔒 Khóa` (2Hz) + Che khuất Z-Buffer + `.arroom` v2:**
   - **`🔒 Khóa`** đứng im hoàn toàn (`±11cm` deadband, không quét sang góc chưa quét, chỉ cập nhật sau 3 nhịp liên tiếp `~1.5s` khi có vật thật đổi); **`🧹 Xóa lưới`** trong **`💾 Phòng`**; tự cập nhật OTA (`🔄` ➔ `⬇ #14`).

---

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
