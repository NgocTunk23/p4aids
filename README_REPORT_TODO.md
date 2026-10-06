# HƯỚNG DẪN VIẾT TIẾP BÁO CÁO (README_REPORT_TODO.md) CHO PHÚ VÀ TIẾN

File này tóm tắt các nội dung bạn cần trình bày tiếp trong báo cáo, bám sát vào những gì chúng ta đã thực hiện thực tế trong dự án.

YÊU CẦU HIỂU RÕ BẢN CHẤT, ta chỉ sài train_features.csv. Các file khác không có ý nghĩa, k sài

## 1. Khám phá dữ liệu ban đầu (EDA)
**Mục tiêu:** Cho người đọc thấy đặc thù của dữ liệu tiêu thụ điện.
- **Nội dung cần viết:**
    Dựa vào EDA đê nói, dùng file preprocessingforALL zone hay không là như nhau chỉ thêm có 1 chỗ để tính zone thôi
  - Phân tích sự mất cân bằng dữ liệu (Anomaly vs Normal). Rất hiếm dị thường (chỉ khoảng 1-5%).
  - Các phát hiện ban đầu: Nhiễu từ cảm biến, sự phụ thuộc của năng lượng vào thời tiết (nhiệt độ) và thời gian (ngày, đêm, cuối tuần).
- **Hình ảnh minh chứng cần chọn:**
  - Biểu đồ phân phối của nhãn (Pie chart hoặc Bar chart thể hiện sự mất cân bằng).
  - Biểu đồ line chart của một vài tòa nhà cụ thể thể hiện các điểm dị thường (Point anomaly) và dị thường theo chuỗi (Sequential anomaly).

## 2. Tiền xử lý dữ liệu và Áp dụng Zone (Khu vực)
**Mục tiêu:** Giải thích cách làm sạch dữ liệu và lý do phân cụm.
- **Các bước tiền xử lý đã làm:** ĐẦY ĐỦ CÁC BƯỚC TRONG TIỀN XỬ LÝ(BÊN DƯỚI CHỈ NÊU KHÁI QUÁT)
  - Xử lý Missing values (điền khuyết) bằng các phương pháp nội suy hoặc loại bỏ nếu cần.
  - Tạo các đặc trưng mới (Feature Engineering): Các đặc trưng thời gian (Giờ, Thứ, Tháng), các đặc trưng thay đổi giá trị (Value-change features như lag, difference).
  - Loại bỏ các đặc trưng không cần thiết và chuẩn hóa dữ liệu.
- **Áp dụng Zone (Phân cụm khu vực):**
  - **Lý do:** Khí hậu, múi giờ và đặc điểm thời tiết ở từng khu vực (Site/Zone) là khác nhau, dẫn đến hành vi tiêu thụ điện khác nhau. Việc gộp chung tất cả sẽ làm mô hình bị nhiễu.
  - **Cách làm:** Chia nhỏ và xử lý đặc trưng theo `site_id` hoặc cụm `Zone`. Tạo bộ dữ liệu `train_preprocessedzone.parquet` và `test_preprocessedzone.parquet`.
- **Hình ảnh/Minh chứng:**
  - Bảng so sánh hoặc biểu đồ cho thấy phân bố nhiệt độ/năng lượng khác nhau giữa 2 Zone bất kỳ.

## 3. Ý nghĩa của các file trong `reportmdcreated/`
Bạn cần đưa các kết quả từ các file này vào báo cáo để làm minh chứng cho quá trình thử nghiệm:
- **`advanced_ensemble_report_old.md` / `new.md`:** 
  - Ghi nhận lại các thử nghiệm Ensemble (XGBoost + LightGBM + CatBoost).
  - Chứng minh rằng việc kết hợp các mô hình (Weighted Blending, Soft Voting) mang lại kết quả F1-score cao hơn mô hình đơn lẻ.
  - Sự khác biệt giữa `old` và `new` thể hiện quá trình cải tiến trọng số và ngưỡng.
- **`final_comprehensive_report_old.md` / `new.md`:** 
  - Bảng tổng hợp toàn diện mọi giai đoạn từ Tiền xử lý (Baseline vs Zone) đến Thresholding.
  - Đây là "chìa khóa" để bạn đưa ra bảng kết quả cuối cùng trong báo cáo, chứng minh Zone + Ensemble + Dynamic Threshold là chiến thuật tốt nhất.

## 4. Huấn luyện, Tinh chỉnh và Tối ưu Mô hình (Modeling & Tuning)
**Mục tiêu:** Chứng minh bạn không chỉ "chạy code" mà có sự tối ưu bài bản.
- **Chiến lược Huấn luyện & Lý do chọn thuật toán:**
  - Sử dụng các thuật toán dạng Cây mạnh nhất (Tree-based) thuộc họ Gradient Boosting: XGBoost, LightGBM, CatBoost. Lý do: Xử lý tốt dữ liệu dạng bảng và đặc biệt nhạy bén với dữ liệu mất cân bằng.
  - **Vấn đề của các mô hình Tree truyền thống (như Random Forest, Decision Tree):** Ban đầu có xem xét tính khả thi của các dạng Tree truyền thống. Tuy nhiên, các thuật toán này có nhược điểm chí mạng với bộ dữ liệu lớn: **thời gian huấn luyện (training time) quá lâu**, ngốn nhiều tài nguyên, mà **hiệu suất (F1-score) lại không cao** và không thể sánh bằng sức mạnh của 2 thuật toán hàng đầu là **XGBoost và LightGBM**. Do đó, báo cáo sẽ xoáy sâu vào các mô hình Boosting.
  - Ban đầu thử nghiệm trên tập chia 80/20 để đánh giá nhanh, sau đó **train 100% dữ liệu** cho file nộp bài (submission) để tận dụng tối đa dữ liệu. KHÔNG YÊU CẦU VIẾT DÔ NHƯNG PHẢI HIỂU LÀ CHỈ CÓ KHI NỘP TRÊN KAGGLE MỚI TRAIN FULL NÊN ẢNH image là lúc nộp
  CÒN LẠI LÀ TOÀN TỰ TRAIN bằng tiền xử lý
- **Tinh chỉnh siêu tham số (Hyperparameter Tuning):** 2 file run_tunning và tune_zone là dùng để tìm tham số tối ưu
  - Đã chạy tìm kiếm tham số tối ưu (ví dụ: `learning_rate`, `max_depth`, `n_estimators`, `num_leaves`) riêng biệt cho cụm **Baseline** và cụm **Zone**.
  - **Minh chứng:** Đưa ra các bảng siêu tham số tốt nhất của từng mô hình (lấy từ các file log hoặc script).
- **Tối ưu Ngưỡng (Threshold Optimization) và Ensemble:**
  - **Ngưỡng tối ưu (Optimal Threshold):** Trong bài toán phát hiện dị thường có sự mất cân bằng lớp cực lớn, việc dùng ngưỡng dự đoán mặc định (`0.5`) thường thất bại (mô hình sẽ có xu hướng dự đoán lớp đa số là Normal, làm sót dị thường). **Việc áp dụng ngưỡng tối ưu là hoàn toàn khả thi và đem lại hiệu quả thực tế**. Bằng cách chạy thuật toán dò tìm, chúng ta tìm ra điểm cắt (cut-off) giúp hàm F1-score đạt giá trị cực đại trên tập Validation. 
  - **Cải tiến bằng Ngưỡng động (Dynamic Threshold) với 5-fold OOF:** Để tránh hiện tượng Overfit khi tối ưu ngưỡng, chúng ta còn áp dụng kỹ thuật 5-fold OOF (Out-of-Fold) để sinh ra ngưỡng động phù hợp với từng phân phối dự đoán cụ thể của cụm dữ liệu. OOF áp dụng khi train full để test KAGGLE còn đa ngưỡng nghĩa là mỗi một khu vực 1 ngược riêng, nếu check khu vực A có ngưỡng là 0.1 thì những đứa vượt ngưỡng khả  năng là anomaly cao
  - **Ensemble Trọng số:** Tìm ra công thức kết hợp tối ưu (ví dụ Baseline là `[0.83 XGB, 0.00 LGB, 0.17 CAT]`).
- **Nguồn tài liệu tham khảo cho các phương pháp Tối ưu (Cơ sở lý thuyết):**
  - **Kỹ thuật Ensemble bằng Blending Weight:** Được tham khảo từ nguyên lý kết hợp mô hình trong bài báo *"Ensemble Selection from Libraries of Models"* (Caruana et al., ICML 2004) và đúc kết từ các giải pháp chiến thắng trong bài toán dự báo năng lượng tòa nhà *ASHRAE Great Energy Predictor III (Kaggle)*. Việc tối ưu hóa trọng số (Weighted Blending) thay vì trung bình cộng đơn thuần giúp mô hình tận dụng tối đa thế mạnh riêng của từng thuật toán (XGBoost, LightGBM, CatBoost) trên từng phân cụm dữ liệu.
  - **Kỹ thuật Ngưỡng động (Dynamic Threshold) cho từng tòa nhà/Zone:** Tham khảo cơ sở lý thuyết từ phương pháp "Nonparametric Dynamic Thresholding" trong bài báo nổi tiếng *"Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding"* (Hundman et al., KDD 2018). Việc áp dụng ngưỡng động giải quyết triệt để vấn đề: mỗi tòa nhà/khu vực có hành vi tiêu thụ điện và mức độ nhiễu khác nhau, nên việc dùng một ngưỡng cứng `0.5` cho toàn bộ là không hợp lý. Ngưỡng cần được tinh chỉnh động theo phân phối xác suất dự đoán của từng cụm.
- **Hình ảnh/Minh chứng:**
  - Bảng so sánh F1-score/Logloss trước và sau khi Tuning.
  - Bảng thể hiện sự tăng tiến điểm số khi đi từ: Single Model -> Ensemble -> Zone Ensemble -> Zone Ensemble + Dynamic Threshold.

  NGOÀI RA CẦN NÓI TỚI KĨ THUẬT STACK, SOFT VOTING(TRUNG BÌNH CỘNG), BLENDING WEIGHT


  GHI ĐẦY ĐỦ CÁC PHẦN CÒN LẠI DÀNH CHO ML CỦA BÁO CÁO, SAU ĐÓ DL BỔ SUNG---> LÀM CODE GIAO DIỆN + SLIDE
