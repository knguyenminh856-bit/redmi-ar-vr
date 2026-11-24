# Redmi Note 14 4G — Rust AR/VR Passthrough Demo

Ứng dụng thực tế hỗn hợp (Passthrough AR/VR) siêu nhẹ viết bằng **Rust + OpenGL ES 3.0 (`glow`) + ARCore / Camera2** tối ưu riêng cho **Xiaomi Redmi Note 14 4G** (`arm64-v8a`, GPU Mali-G57 MC2, màn hình AMOLED 120Hz `2400×1080`).

## Tính năng chính

1. **Nhìn xuyên thấu chống chóng mặt (Anti-Motion-Sickness Passthrough):**
   - Dùng `GL_TEXTURE_EXTERNAL_OES` đẩy trực tiếp khung hình camera vào GPU (**Zero-copy**).
   - Hiển thị camera trong cửa sổ **tỷ lệ góc 1:1** ở khoảng cách hội tụ `2.5m` trên nền đen AMOLED (`#000000`), giúp đồ vật ngoài đời đúng kích thước mắt thật và không gây buồn nôn khi xoay đầu.
   - **120Hz Async Gyro Timewarp:** Tách vòng lặp `ArSession.update()` (~32Hz) khỏi vòng lặp vẽ `120Hz`, sử dụng cảm biến `TYPE_GAME_ROTATION_VECTOR` để bù xoay đầu tức thời giữa các khung hình camera.
2. **2 khối 3D nổi lơ lửng (Stereoscopic 3D):**
   - 1 khối **Lập phương** đa sắc (`1.25m`) và 1 khối **Bát diện / Tinh thể** nhấp nhô (`1.65m`) tách 2 mắt chuẩn `IPD = 64mm`, neo cố định trong không gian 6DoF qua ARCore (tự động chuyển sang `Camera2` + Gyroscope 120Hz 3DoF nếu điện thoại chưa cài Google Play Services for AR).
3. **Nhận diện bàn tay 5 ngón & Cử chỉ thông minh (MediaPipe + Rust GLES):**
   - Chạy ngầm MediaPipe `HandLandmarker` không làm nghẽn khung hình 120Hz, tự động vẽ **5 đường xương nối 21 khớp từ cổ tay ra 5 ngón tay + lòng bàn tay** khớp lên hình camera 2 mắt.
   - Nhận diện các thao tác tay theo tọa độ 3D thực và hiện **Popup nổi 2 mắt** khi phát hiện: **✊ Nắm đấm**, **🤏 Chụm ngón cái**, **👉 Hình cây súng**, **✌ Chữ V** (xòe bàn tay để đặt lại).

---

## Tải file `.apk` tự động từ GitHub Actions

Mỗi khi push code lên nhánh `main` (hoặc chạy tay từ tab **Actions**), GitHub Actions sẽ tự động biên dịch thư viện Rust `arm64-v8a` và đóng gói file **`RedmiNote14-ARVR-debug.apk`**:

1. **Cách nhanh nhất trên điện thoại:** Vào mục **Releases** của repository này → chọn bản **`Latest APK Build (Redmi Note 14 4G)`** (`latest-apk`) → tải trực tiếp file `RedmiNote14-ARVR-debug.apk` và bấm cài đặt.
2. **Hoặc tải từ Workflow Artifacts:** Vào tab **Actions** → chọn lượt build xanh gần nhất → tải artifact `RedmiNote14-ARVR-APK` và giải nén lấy file `.apk`.

---

## Tinh chỉnh hộp VR & sử dụng trên Redmi Note 14 4G

1. Mở app, cấp quyền **Camera** (và nên bật tần số quét **120Hz** trong Cài đặt hiển thị của máy nếu đang khóa 60Hz).
2. Trước khi nhét máy vào hộp VR, chạm nút **`⚙`** ở góc trên bên phải để mở bảng chỉnh nhanh (tự động lưu):
   - **FOV Scale (`50% – 100%`, mặc định `78%`):** Thu/phóng khung hình camera sao cho kích thước vật thật qua thấu kính hộp VR khớp với mắt nhìn tự nhiên.
   - **IPD (`56 – 72 mm`, mặc định `64 mm`):** Chỉnh khoảng cách 2 mắt cho đỡ mỏi mắt.
   - **Đặt lại khối 3D sau 3 giây:** Bấm nút này rồi nhét nhanh điện thoại vào hộp VR; sau 3 giây, 2 khối 3D sẽ tự neo ngay trước tầm nhìn của bạn.
3. **Mẹo khi đã đóng nắp hộp VR:** Bấm nút cứng **Tăng âm lượng (`Volume Up`)** bên hông điện thoại bất kỳ lúc nào để đặt lại vị trí 2 khối 3D ngay trước mặt!
