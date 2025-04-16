# Nhận Diện Khuôn Mặt Dự Đoán Tuổi và Giới Tính
## Tổng Quan
Dự án này triển khai một mô hình học sâu để nhận diện khuôn mặt, dự đoán tuổi và giới tính của cá nhân dựa trên hình ảnh khuôn mặt. 
Mô hình sử dụng kiến trúc MobileNetV2 được huấn luyện trước để trích xuất đặc trưng, kết hợp với một mạng nơ-ron tùy chỉnh để phân loại giới tính (nam/nữ) và các nhóm tuổi (được chia thành 9 khoảng). 
Dữ liệu được tổ chức thành các thư mục theo độ tuổi và giới tính, và mô hình được huấn luyện bằng TensorFlow/Keras trên môi trường hỗ trợ GPU. Bộ dữ liệu được lấy từ /kaggle/input/dataimage/dataprojectfinal
, bao gồm các hình ảnh khuôn mặt được sắp xếp vào các thư mục theo khoảng tuổi và giới tính. Các khoảng tuổi bao gồm: 0-3, 4-10, 11-20, 21-30, 31-40, 41-50, 51-60, 61-70, 71+ tuổi
Nhãn giới tính: Nam (0), Nữ (1), hoặc Không xác định (-1 cho nhóm tuổi 0-3)
## Dữ liệu được tiền xử lý thành một DataFrame với các cột:
- image_path: Đường dẫn đến tệp hình ảnh
- gender_label: Nhãn giới tính (0, 1, hoặc -1)
- age_label: Nhãn nhóm tuổi (ví dụ: 0, 4, 11, 21,...)
## Thống Kê Dữ Liệu
- Phân bố Tuổi: Được trực quan hóa bằng biểu đồ cột để hiển thị số lượng hình ảnh theo từng nhóm tuổi.
- Phân bố Giới Tính: Đếm số lượng hình ảnh cho các danh mục nam, nữ và không xác định.
## Phương Pháp
### Tiền Xử Lý Dữ Liệu
- Tải Hình Ảnh: Hình ảnh được tải từ thư mục dữ liệu và ánh xạ với nhãn tuổi và giới tính tương ứng.
- Trích Xuất Đặc Trưng: Hình ảnh được tiền xử lý bằng hàm preprocess_input của MobileNetV2.
Đặc trưng được trích xuất theo batch bằng mô hình MobileNetV2 đã huấn luyện trước (loại bỏ lớp đầu ra). Các đặc trưng được làm phẳng để đưa vào mô hình tùy chỉnh.
### Kiến Trúc Mô Hình: Mô hình bao gồm:
- Mô Hình Cơ Sở: MobileNetV2 (được huấn luyện trước trên ImageNet) để trích xuất đặc trưng.
- Mạng Nơ-ron Tùy Chỉnh:
+ Lớp đầu vào nhận đặc trưng đã làm phẳng từ MobileNetV2.
+ Nhiều lớp dense (16, 32, 64, 128, 256 đơn vị) với: Hàm kích hoạt ReLU, Chính quy hóa L2 (0.01), Dropout (0.5) để ngăn chặn overfitting
+ Hai lớp đầu ra:
  gender_output: Lớp softmax cho phân loại giới tính nhị phân (2 lớp).
  age_output: Lớp softmax cho phân loại nhóm tuổi (9 lớp).
### Huấn Luyện
- Bộ Tối Ưu: Adam với tốc độ học ban đầu là 0.001.
- Hàm Mất Mát: categorical_crossentropy cho cả đầu ra giới tính và tuổi.
- Độ Đo: Độ chính xác (accuracy) cho cả dự đoán giới tính và tuổi.
- Callback:
  EarlyStopping: Dừng huấn luyện nếu mất mát trên tập xác thực không cải thiện trong 5 epoch, khôi phục trọng số tốt nhất.
  ReduceLROnPlateau: Giảm tốc độ học xuống 0.2 lần nếu mất mát trên tập xác thực không cải thiện trong 3 epoch (tốc độ học tối thiểu: 0.00001).
- Chi Tiết Huấn Luyện:
  Kích thước batch: 32
  Số epoch: 10
  Tập huấn luyện và xác thực được chia bằng train_test_split.
### Đánh Giá
- Mô hình xuất ra xác suất cho dự đoán giới tính và tuổi.
