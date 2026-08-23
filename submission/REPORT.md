# Báo Cáo Thực Hành Lab 21: CI/CD cho AI Systems

## 1. Bộ Siêu Tham Số Đã Chọn
Dựa trên kết quả so sánh 3 lần chạy trong MLflow UI với các bộ tham số khác nhau:
- Lần 1: `n_estimators=50`, `max_depth=3`, `min_samples_split=2` (Accuracy: 0.5580, F1: 0.5185)
- Lần 2: `n_estimators=100`, `max_depth=5`, `min_samples_split=5` (Accuracy: 0.5680, F1: 0.5573)
- Lần 3: `n_estimators=200`, `max_depth=10`, `min_samples_split=5` (Accuracy: 0.6440, F1: 0.6417)

**Lý do chọn Lần 3 (`n_estimators=200`, `max_depth=10`, `min_samples_split=5`):** 
Bộ siêu tham số này đạt kết quả tốt nhất trên tập eval (Accuracy 0.6440). Việc tăng số lượng cây quyết định (`n_estimators`) giúp mô hình học được nhiều quy luật và ensemble mượt mà hơn. Độ sâu `max_depth=10` kết hợp `min_samples_split=5` cho phép mô hình học cấu trúc phức tạp mà vẫn đủ cẩn thận để tránh overfitting như khi không giới hạn độ sâu.

## 2. So Sánh Hiệu Suất Khi Tăng Dữ Liệu
- Khi huấn luyện trên **2.998 mẫu (Phase 1)**: Accuracy: `0.6440` | F1-Score: `0.6417`
- Khi huấn luyện trên **5.996 mẫu (Phase 1 + 2)**: Accuracy: `0.6620` | F1-Score: `0.6583`

**Đánh giá:** Rõ ràng, việc bổ sung thêm dữ liệu huấn luyện đã giúp mô hình cải thiện hiệu suất rõ rệt. Quá trình Continuous Training tự động hoạt động mang lại giá trị lớn khi Accuracy tăng lên khoảng 1.8% và F1-Score cũng tăng đồng đều, cho thấy mô hình không bị thiên lệch (bias) mà đã tổng quát hóa (generalize) tốt hơn.

## 3. Khó Khăn Gặp Phải Và Cách Giải Quyết
- **Khó khăn 1:** Cài đặt phụ thuộc (dependencies) trong môi trường CI/CD mất nhiều thời gian và đôi khi dẫn đến xung đột với các phiên bản package.
  - **Cách giải quyết:** Ghim chặt các phiên bản trong `requirements.txt` (vd: `pandas==2.2.2`, `scikit-learn==1.4.2`) để đảm bảo pipeline luôn ổn định và nhất quán giữa local và runner.
- **Khó khăn 2:** Xác thực Cloud Storage (GCP) để upload Model.
  - **Cách giải quyết:** Trong job Train, cần tạo file `/tmp/sa-key.json` từ GitHub Secrets, sau đó cấu hình biến môi trường `GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json` để script Python có thể gọi SDK upload thành công.
- **Khó khăn 3:** Pass dữ liệu metrics giữa các Jobs (từ Train sang Eval Gate).
  - **Cách giải quyết:** Sử dụng Output Parameters. Tại job Train, tôi đọc giá trị bằng đoạn script `acc=$(python -c "import json; print(json.load(open('outputs/metrics.json'))['accuracy'])")` và gán thông qua `$GITHUB_OUTPUT`. Sau đó, lấy `$acc` bên job Eval thông qua `${{ needs.train.outputs.accuracy }}`.
