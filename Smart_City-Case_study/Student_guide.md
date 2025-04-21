# Hướng dẫn chi tiết cho sinh viên: Giải thích mã `traffic_light_violation.py`

## 1. Mục tiêu của bài tập

- **Phát hiện và theo dõi (tracking) ô tô** trên video sử dụng YOLOv8n.
- **Phát hiện trạng thái đèn giao thông** (đỏ/xanh/vàng/tắt) bằng model `best_traffic_nano_yolo.pt`.
- **Xác định** khi ô tô vượt vạch trong lúc đèn đỏ và **lưu ảnh** vi phạm.
- **Minh hoạ kết quả**: vẽ bounding box, trạng thái đèn, đếm số vi phạm, hiển thị thông báo vi phạm.

---

## 2. Yêu cầu môi trường và thư viện

- **Python 3.12** hoặc cao hơn.
- Thư viện:
  ```bash
  pip install ultralytics opencv-python numpy
  ```

---

## 3. Cấu trúc chính của script

```text
traffic_light_violation.py        # script chính
hi2.mp4           # video đầu vào
yolov8n.pt        # YOLOv8n COCO model (detect ô tô)
best_traffic_nano_yolo.pt  # model detect đèn
vi_pham/          # thư mục tự động tạo để lưu ảnh vi phạm
```

---

## 4. Vẽ vạch kiểm soát (line)

1. **Biến toàn cục** `line_pts = []` chứa tối đa 2 điểm do người dùng click.
2. **Hàm** `mouse_callback(event, x, y, flags, param)`:
   - Bắt sự kiện `cv2.EVENT_LBUTTONDOWN` (nhấp trái chuột).
   - Nếu `len(line_pts) < 2`, thêm `(x,y)` vào `line_pts` và in ra console.
3. **Chọn vạch**:
   - Đọc frame đầu tiên từ video.
   - Hiển thị lên cửa sổ `Select Line`.
   - Gọi `cv2.setMouseCallback` với `mouse_callback`.
   - Vẽ:
     ```python
     if len(line_pts) >= 1:
         cv2.circle(frame, line_pts[0], 5, (0,255,0), -1)
     if len(line_pts) == 2:
         cv2.line(frame, line_pts[0], line_pts[1], (0,255,0), 2)
     ```
   - Nhấn **`s`** để xác nhận 2 điểm, **`q`** để hủy.

---

## 5. Tải các model YOLO

- **Ô tô (car\_model)**:
  ```python
  car_model = YOLO('yolov8n.pt')  # COCO class 2 = car
  ```
- **Đèn giao thông (tl\_model)**:
  ```python
  tl_model = YOLO('best_traffic_nano_yolo.pt')
  # class 0:red, 1:green, 2:off, 3:yellow
  ```

---

## 6. Hàm phụ trợ: `side_of_line`

```python
def side_of_line(pt, p1, p2):
    # Tính giá trị vecto (x2-x1, y2-y1) ⋅ (pt - p1) theo dạng 2D cross
    return (x2-x1)*(pt[1]-y1) - (y2-y1)*(pt[0]-x1)
```

- Kết quả >0: `pt` nằm về một phía của line, <0: phía kia, =0: trên line.
- Dùng để kiểm tra **dấu** của điểm trước và sau:
  ```python
  s_prev = side_of_line(prev_pt, p1, p2)
  s_curr = side_of_line(curr_pt, p1, p2)
  if s_prev * s_curr < 0:
      # pt đã đi qua line
  ```

---

## 7. Cấu trúc lưu trữ lịch sử (track\_history)

Mỗi `track_id` là khoá, giá trị là dict:

```python
track_history[tid] = {
    'pt': (cx,cy),           # bottom-center từ frame trước
    'crossed': False,        # đã check crossing chưa
    'violation': False,      # đã vi phạm chưa
    'violation_time': None   # timestamp vi phạm
}
```

- `cx, cy = bottom center = ((x1+x2)//2, y2)`
- Sau mỗi frame, cập nhật `rec['pt'] = (cx, cy)`.

---

## 8. Vòng lặp chính (tracking & detection)

```python
for result in car_model.track(source=VIDEO_SOURCE,
                              conf=0.5, iou=0.5,
                              classes=[2], persist=True,
                              stream=True):
    frame = result.orig_img.copy()
    frame_count += 1
    # 1) Vẽ vạch
    # 2) Detect đèn giao thông -> light_label
    # 3) Duyệt result.boxes để track từng ô tô
    #    - lấy bottom-center (cx,cy)
    #    - nếu chưa crossed, compute s_prev, s_curr
    #      + nếu crossed và light_label=='red': set violation
    #      + lưu violation_time, lưu ảnh crop vào vi_pham/
    #    - update crossed và pt
    #    - chọn box_color dựa vào violation flag
    #    - vẽ box, ID, tâm, và chữ "VI PHAM" nếu trong 1s đầu sau vi phạm
    # 4) Đếm violations và hiển thị
    # 5) Hiển thị frame và xử lý phím 'q'
```

### 8.1. Phát hiện đèn giao thông

- Chạy `tl_model(frame, conf=0.3)[0]` để lấy kết quả đầu tiên.
- Duyệt `tl_res.boxes`:
  - Lấy `x1,y1,x2,y2`, `cls_id`, `conf`.
  - `name = tl_model.model.names[cls_id]`.
  - Vẽ box với màu:
    - `green` → (0,255,0)
    - `red`   → (0,0,255)
    - `yellow`→ (255,255,0)
  - Chọn detection có `conf` cao nhất làm `light_label`.

### 8.2. Xử lý mỗi ô tô

1. **Lấy ID**: `tid = int(box.id.cpu().item())` để tránh DeprecationWarning.
2. **Tính bottom-center**: `cx = (x1+x2)//2`, `cy = y2`.
3. **Khởi tạo** record nếu `tid` mới.
4. **Kiểm tra Crossing** nếu `not rec['crossed']`:
   ```python
   if s_prev * s_curr < 0:
       if light_label == 'red':
           rec['violation'] = True
           rec['violation_time'] = time.time()
           # lưu crop ảnh
       rec['crossed'] = True
   ```
5. **Cập nhật** `rec['pt'] = (cx, cy)`.
6. **Chọn màu** `box_color`:
   - `violation=True` → đỏ `(0,0,255)`
   - ngược lại → xanh `(255,0,0)`
7. **Vẽ** box, ID, tâm: `cv2.rectangle`, `cv2.putText`, `cv2.circle`.
8. **Hiển thị "VI PHAM"**:
   ```python
   if rec['violation'] and time.time() - rec['violation_time'] <= 1.0:
       cv2.putText(frame, "VI PHAM", ...)
   ```

### 8.3. Đếm và hiển thị số vi phạm

```python
violation_count = sum(v['violation'] for v in track_history.values())
cv2.putText(frame, f"Violations: {violation_count}", (10,60), ...)
```

---

## 9. Thoát chương trình

- Nhấn `q` để dừng vòng lặp và đóng tất cả cửa sổ.

---

## 10. Gợi ý tùy chỉnh

- **Ngưỡng conf/iou**: thay trong `car_model.track` và `tl_model(...)`.
- **Giới hạn vùng phát hiện đèn**: crop chỉ khu vực giao lộ — tăng tốc.
- **Thêm logging**: lưu file CSV, video output…
- **Thông báo thời gian thực**: phát âm thanh, gửi HTTP request …

---

**Mở rộng.*
- **Thay đổi đầu vào của chương trình**: Chuyển việc xử lý Video sang xử lý Video Stream từ Esp32-camera. 
- **Tích hợp nhận dạng biển số xe (OCR)**: sau khi crop ảnh vi phạm, sử dụng thư viện OCR (ví dụ: Tesseract hoặc OpenALPR) để đọc và trích xuất biển số, lưu biển số kèm timestamp và file ảnh.
- **Lưu trữ vào cơ sở dữ liệu**: kết nối SQLite hoặc MySQL để lưu thông tin vi phạm (biển số, trạng thái đèn, vị trí vạch, đường dẫn ảnh), phục vụ tra cứu và thống kê.
- **Xây dựng giao diện giám sát Web**: dùng Flask hoặc Django để hiển thị video live, danh sách vi phạm, biểu đồ thống kê số lần vi phạm theo thời gian.
- **Báo cáo tự động**: tổng hợp dữ liệu vi phạm hàng ngày/tuần/tháng, xuất file CSV, Excel hoặc PDF tự động và gửi email đến người quản lý.
- **Hệ thống cảnh báo thời gian thực**: tích hợp gửi email, SMS hoặc thông báo push khi có xe vi phạm mới.
- **Mở rộng logic**: lưu vecto chuyển động để tính tốc độ, kết hợp phân loại loại xe (ô tô con, xe tải) để phân loại vi phạm.

*Created by Nguyen Van Nhan*
[![AIoTLab](https://img.shields.io/badge/AIoTLab-green?style=for-the-badge)](https://www.facebook.com/DNUAIoTLab)
[![Faculty of Information Technology](https://img.shields.io/badge/Faculty%20of%20Information%20Technology-blue?style=for-the-badge)](https://dainam.edu.vn/vi/khoa-cong-nghe-thong-tin)
[![DaiNam University](https://img.shields.io/badge/DaiNam%20University-orange?style=for-the-badge)](https://dainam.edu.vn)
