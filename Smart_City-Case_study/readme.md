Một dự án ví dụ cho các bài toán nằm trong Thành phố thông minh

# README

## Yêu cầu môi trường

1. **Python 3.12**: Tải và cài đặt từ https://www.python.org/downloads/
2. **Tạo virtual environment** (khuyến nghị):
   ```bash
   python3.12 -m venv myenv
   source myenv/bin/activate   # Linux/Mac
   myenv\Scripts\activate    # Windows PowerShell
   ```

## Cài đặt phụ thuộc

```bash
pip install --upgrade pip
pip install ultralytics opencv-python numpy
```  
(Đảm bảo phiên bản OpenCV và NumPy tương thích với Python 3.12.)

## Cấu trúc thư mục

```
project_root/
├── hi2.mp4                   # video đầu vào (hoặc đổi tên khác)
├── best_traffic_nano_yolo.pt # model detect đèn giao thông
├── yolov8n.pt                # model COCO để detect ôtô
├── vi_pham/                  # (tự động tạo) lưu ảnh vi phạm
└── 2_light.py                # script chính
```  

## Cách chạy

1. **Chỉnh biến** `VIDEO_SOURCE` trong `2_light.py` thành đường dẫn đến video (ví dụ: `r'D:\Source\hi2.mp4'`).
2. Chạy script:
   ```bash
   python 2_light.py
   ```
3. Khi cửa sổ `Select Line` mở lên:
   - Nhấn chuột trái để chọn **2 điểm** xác định vạch kiểm soát.
   - Nhấn **`s`** để xác nhận hoặc **`q`** để hủy.
4. Màn hình `Tracking & Violation` sẽ hiển thị:
   - **Bounding box** ôtô (xanh dương) và box đèn giao thông (đỏ/xanh/vàng).
   - Chữ **`Light: <state>`** (trạng thái đèn) tại góc trên trái.
   - **Số lượng vi phạm** (`Violations: N`).
   - Khi ôtô vượt vạch lúc đèn đỏ: box ôtô chuyển thành **đỏ**, chữ **`VI PHAM`** xuất hiện trong 1 giây và ảnh crop được lưu vào `vi_pham/`.
5. Nhấn **`q`** để thoát.

## Tùy chỉnh

- **Ngưỡng**: thay đổi `conf`, `iou` khi gọi `car_model.track(...)` và `tl_model(...)`.
- **Logic** vi phạm: có thể thay `light_label=='red'` thành điều kiện khác (ví dụ chỉ `yellow` hoặc `off`).
- **Vùng detect đèn**: cắt ROI để tăng tốc độ nếu cần.

---

*Created by Khoa Công nghệ Thông tin*
