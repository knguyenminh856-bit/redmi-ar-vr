# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) tối ưu cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, Helio G99-Ultra, GPU Mali-G57 MC2, AMOLED 120Hz `2400×1080`), viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore Depth + MediaPipe**.

## Tính năng chính

1. **Nhìn xuyên thấu 120Hz chống chóng mặt & tối ưu hiệu năng:**
   - **Zero-Copy OES Camera:** Luồng camera phần cứng đưa thẳng vào GPU ở độ phân giải sắc nét (`≥1080p` GPU, `≤480p` cho AI), tắt MSAA mặc định và giữ GPU Mali-G57 100% rảnh rỗi cho 120Hz.
   - **1:1 Angular Scale + Async Timewarp:** Cửa sổ camera khớp tỷ lệ thật ở `2.5m` trên nền đen AMOLED kèm bù xoay đầu `TYPE_GAME_ROTATION_VECTOR` 120Hz.
   - **Làm mượt 1€ Filter (`OneEuroFilter`):** Làm mượt 21 khớp tay từ MediaPipe CPU (`20Hz`) lên mượt `120Hz`.
2. **Che khuất theo chiều sâu (Hybrid Depth Occlusion):**
   - **ARCore 16-bit Depth API** (`15Hz`, shader `GL_RG8`): Vật thể ảo tự ẩn đi khi bị bàn tay, mặt bàn, cốc nước hoặc đồ vật ngoài đời che phía trước.
   - **Khử Z-buffer bàn tay 120Hz tức thì:** Đưa khung bàn tay & 5 ngón vào Z-buffer để bàn tay luôn che khuất được khối 3D mượt ngay cả trong chế độ Camera2 fallback.
3. **2 khối 3D nổi + Vẽ không gian bằng đầu ngón tay:**
   - **Chế độ Vẽ (`✏ Vẽ: TẮT / 1 NGÓN / 5 NGÓN`):** Vẽ dải neon 3D lơ lửng ngay tại đầu ngón trỏ (khi giơ `☝` / `🤏`) hoặc đồng thời cả 5 đầu ngón tay (nắm đấm `✊` để nhấc bút).
   - Bấm **`🗑 Xóa`** (hoặc dùng combo `✌ ➔ ✊`) để xóa toàn bộ nét vẽ.
4. **8 Cử chỉ & 5 Combo hành động:**
   - **8 cử chỉ:** `✊ Nắm đấm`, `🤏 Chụm ngón`, `👉 Cây súng`, `✌ Chữ V`, `🖐 Xòe tay`, `👍 Thích`, `🤘 Rock`, `☝ Chỉ 1 ngón`.
   - **5 Combo (thực hiện liên tiếp trong `1.65s`):**
     - `🤏 ➔ ✊`: Hút / Đẩy 2 khối 3D lại gần (`0.78m` ↔ `1.25m`).
     - `✊ ➔ 🖐`: Kích hoạt sóng xung kích quay nhanh.
     - `🤏 ➔ 👉`: Đổi bảng màu 2 khối 3D.
     - `✌ ➔ ✊`: Xóa sạch nét vẽ tay trong không gian.
     - `🤘 ➔ 🖐`: Đặt lại tâm khối 3D trước mặt.
5. **Bảng thông số chi tiết của máy (`📊 Thông số`):**
   - Hiển thị đồng bộ cả 2 mắt trong kính VR: tên máy/SoC, GPU OpenGL, tần số màn hình, **Render FPS / ms**, **AI Hand FPS / ms**, khoảng cách tâm Depth (`m`), **tọa độ 3D đầu ngón trỏ (`X, Y, Z`)**, **Pin (%)**, **Nhiệt độ (`°C`)** và **RAM đang dùng (`MB`)**.

---

## Tải file `.apk` tự động từ GitHub Actions

Vào **Releases** → chọn **`Latest APK Build (Redmi Note 14 4G)`** (`latest-apk`) → tải trực tiếp file `RedmiNote14-ARVR-debug.apk` về điện thoại cài đặt.

## Phím điều khiển nhanh khi đã nhét máy vào hộp VR
* **Nút Tăng âm lượng (`Volume Up`):** Đặt lại vị trí 2 khối 3D về ngay trước tầm mắt.
* **Nút Giảm âm lượng (`Volume Down`):** Chuyển chế độ vẽ tay (`TẮT` → `1 NGÓN` → `5 NGÓN`).
