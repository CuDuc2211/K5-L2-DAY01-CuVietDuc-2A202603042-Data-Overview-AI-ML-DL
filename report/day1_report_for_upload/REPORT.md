# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
-Record này nói về cả bức ảnh, không phải một vật thể riêng lẻ.

Vì đây là bài toán phân loại ảnh (Image Classification), mô hình đưa ra nhãn dự đoán cho toàn bộ hình ảnh. Record không chứa tọa độ bounding box để xác định vị trí của từng vật thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
-Danh sách tên lớp không phải do mô hình tự nghĩ ra mà được định nghĩa sẵn trong taxonomy đi kèm với mô hình đã huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
-Cần ghi cả ba để xác định chính xác lớp dự đoán, tránh nhầm lẫn giữa các mô hình hoặc bộ taxonomy khác nhau.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
-Guideline cần quy định rõ cách chọn nhãn khi ảnh chứa nhiều vật thể
6. Vì sao không được lấy `score` của model làm ground truth?
-Vì score chỉ thể hiện mức độ tin cậy của mô hình đối với dự đoán, không chứng minh dự đoán đó chính xác.
Mô hình có thể dự đoán sai ngay cả khi score cao. Ground truth phải được xác định theo guideline và được con người kiểm tra, không thể lấy trực tiếp từ score.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=bowl ; `score`=0.719063 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`=67.51 ; `bbox_height`=42.81
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
-Khung chữ nhật của chiếc bát bắt đầu ở góc trên-trái `(32.65, 342.12)` và kết thúc tại góc dưới-phải `(100.16, 384.93)`. Chiều rộng = `100.16 − 32.65 = 67.51` pixel; chiều cao = `384.93 − 342.12 = 42.81` pixel. Tọa độ gốc `(0,0)` nằm ở góc trên-trái ảnh; `x` tăng sang phải và `y` tăng xuống dưới.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
-Chọn `0.35` và `0.60`. Ở ngưỡng **0.35**, JSON lưu **11 prediction** của `kitchen`. Trong 11 prediction này, có **6 prediction** có `score >= 0.60`, nên nếu lọc chính các record đã xuất theo ngưỡng 0.60 thì còn **6 prediction**.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
-Reviewer phải rà soát nhiều khung hơn, đặc biệt các prediction score thấp: kiểm tra vật có thật không, lớp đúng không, khung có ôm đúng vật không, có hộp trùng hay bỏ sót vật không; yêu cầu annotator sửa hoặc bỏ nhãn sai nếu cần.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
-Khi vật thể nhìn thấy rõ, vẽ **một bounding box hình chữ nhật ôm sát phần nhìn thấy của vật thể**, không lấy thêm nhiều nền; với vật thể bị che/cắt mép phải áp dụng cùng một quy tắc đã định trước trong guideline. Không tự suy đoán phần bị khuất nếu guideline chưa cho phép.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
-Ở mép trái ảnh `kitchen` có một phần người bị cắt khỏi khung hình (mô hình tạo record `person`, score `0.610931`, `bbox_xyxy = [0.12, 263.18, 61.55, 310.78]`). Đây là trường hợp **cắt mép ảnh**. Nên đưa thành quy tắc chung trong guideline: nếu đủ bằng chứng nhận ra vật thể, chỉ gán box cho phần nhìn thấy trong ảnh theo chuẩn của dự án; nếu không xác định được đối tượng hoặc guideline chưa nói rõ thì **escalation tới Lab Coach**, không tự vẽ phần ngoài ảnh.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-002 ; `class_name`=bowl ; `score`=0.735744 ; số điểm polygon=`polygon_point_count`=67 ; 3–5 điểm đầu của `polygon_xy`=[[53.0, 344.0], [52.0, 345.0], [50.0, 345.0], [49.0, 346.0], [48.0, 346.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
-Bounding box chỉ xác định hình chữ nhật bao vật thể, có thể chứa cả nền; polygon biểu diễn đường biên chi tiết bằng nhiều điểm `(x, y)`, giúp mô tả phần diện tích thực tế của từng vật thể, kể cả hình dạng không đều.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Đây là mã phân biệt **từng đối tượng được dự đoán trong ảnh**, ví dụ `kitchen-002` là một instance của lớp `bowl`. Nhiều bát có thể cùng `class_id = 45` nhưng có `instance_id` khác nhau.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
-** Polygon/mask phải bám sát **ranh giới phần vật thể nhìn thấy**, không lấn sang nền hoặc vật thể kề bên; ở vùng bị che, không tự tưởng tượng hình dạng sau vật chắn trừ khi guideline có quy định rõ.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
-** Ở phía dưới-trái ảnh `kitchen`, một chiếc bát nằm sát mép trái ảnh và nằm trên mặt bàn; record `kitchen-006` (`bowl`, score `0.528201`) có box bắt đầu ở `x = 0.11`, cho thấy vật thể gần như chạm biên ảnh. Cần phân biệt biên chiếc bát với mặt bàn và giới hạn ảnh. **Xử lý theo guideline** bằng cách chỉ vẽ ranh giới quan sát được, dừng ở mép ảnh; nếu không phân biệt được biên do chồng lấp/đường biên mờ hoặc hướng dẫn chưa rõ thì **escalation** cho Lab Coach.
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn lớp đại diện cho **toàn ảnh** theo taxonomy/guideline. | Ảnh `traffic` có nhiều phương tiện nhưng top-1 lại là `cab` (`score = 0.510915`); khó quyết định nhãn đại diện cho cả ảnh. | Quan sát toàn ảnh, chọn nhãn theo guideline; báo ca khó. | Đối chiếu nhãn với nội dung toàn ảnh, taxonomy và quy tắc chọn chủ đề chính. |
| Phát hiện vật thể | Mỗi vật thể cần gán một **class + bounding box** (đơn vị nhãn là từng hộp). | Ảnh `kitchen` có `person` ở mép trái chỉ lộ một phần (`score = 0.610931`, box bắt đầu ở `x = 0.12`); cần quy định box cho vật bị cắt mép. | Gán lớp, vẽ box sát phần vật thể nhìn thấy theo guideline, ghi ca mơ hồ. | Kiểm tra lớp, vị trí/kích thước hộp, vật bỏ sót, hộp sai hoặc trùng và quy tắc với vật cắt mép. |
| Instance segmentation | Mỗi vật thể cần gán **class + mask/polygon riêng** (đơn vị nhãn là từng instance). | Bát bên trái (`kitchen-006`, `bowl`, `score = 0.528201`) sát biên ảnh và tiếp xúc khu vực bàn; dễ nhầm ranh giới mask. | Vẽ mask theo đường biên quan sát được, phân biệt bát với bàn, chuyển ca khó khi cần. | Kiểm tra polygon có lấn nền/vật khác, thiếu vùng nhìn thấy, vẽ vượt biên ảnh hay trộn nhiều instance. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
-Chỉ sử dụng ảnh COCO công khai được cấp cho bài lab, không tự thay bằng ảnh cá nhân/dữ liệu nội bộ, và không đưa thông tin cá nhân hoặc dữ liệu nhạy cảm vào báo cáo/output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
-** Dừng sử dụng và báo **Lab Coach hoặc giảng viên** để được hướng dẫn; không tự xử lý hoặc phát tán dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
