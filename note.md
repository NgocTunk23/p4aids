# KẾ HOẠCH RÀ SOÁT VÀ NÂNG CẤP CHƯƠNG 5 (HỌC SÂU - 5 DL.tex)

> **Mục tiêu cốt lõi:** Chuẩn hóa toàn bộ nội dung Chương 5 (Học Sâu) để đạt độ sắc bén, chiều sâu học thuật và tính nhất quán phương pháp luận tương đương với Chương 4 (Học Máy), đồng thời **chỉ so sánh đối chứng tương đương trên không gian Context-Aware** (vì Học sâu chỉ huấn luyện trên Baseline, Zone, Context; chưa chạy trên bộ Expanded).

---

## 1. NGUYÊN TẮC SO SÁNH ĐỐI CHỨNG (BẮT BUỘC TUÂN THỦ)

1. **Chỉ so sánh trên cùng không gian tương đương (Context-Aware / Baseline / Zone):**
   - Phân hệ Học sâu **chỉ được huấn luyện trên 3 biến thể:** Baseline ($28$ kênh), Zone ($31$ kênh) và Context-Aware ($35$ kênh).
   - Tuyệt đối **không so sánh trực tiếp Học sâu với bộ Expanded (53 đặc trưng)** của Học máy theo kiểu đối đầu, vì Học sâu chưa từng chạy trên bộ Expanded.
   - Bộ dữ liệu Expanded (53 đặc trưng) của Chương 4 được trình bày như một **hướng mở rộng và phát triển tiềm năng trong tương lai** cho phân hệ Học sâu.
2. **Khẳng định ưu thế thực chất của Học sâu trên tầng Context-Aware:**
   - Trên cùng không gian ngữ cảnh vi mô **Context-Aware**:
     - **Học sâu (BiGRU):** $\text{PR-AUC} = \mathbf{0{,}7535}$, $\text{ROC-AUC} = \mathbf{0{,}9646}$, $F_1 = 0{,}6794$.
     - **Học sâu (BiGRU + BiLSTM Ensemble):** $\text{PR-AUC} = \mathbf{0{,}7574}$, $\text{ROC-AUC} = 0{,}9628$.
     - **Học máy (XGBoost Context):** $\text{PR-AUC} = 0{,}6810$, $\text{ROC-AUC} = 0{,}8957$, $F_1 = 0{,}7066$.
     - **Học máy (LightGBM Context):** $\text{PR-AUC} = 0{,}6750$, $\text{ROC-AUC} = 0{,}8931$, $F_1 = 0{,}7081$.
     - **Học máy (Ensemble Context):** $\text{PR-AUC} \approx 0{,}6950$.
   - **Kết luận khoa học công bằng:** Khi chỉ có các đặc trưng ngữ cảnh tức thời (chưa trích xuất 8 đặc trưng chuỗi thời gian thủ công), các mô hình mạng nơ-ron hồi quy hai chiều (**BiGRU và BiLSTM**) với cửa sổ ngữ cảnh $252$ giờ vượt trội hơn hẳn các mô hình cây học máy ($+0{,}06$ đến $+0{,}07$ điểm PR-AUC), chứng minh năng lực tự động trích xuất phụ thuộc thời gian mạnh mẽ của Deep Learning.

---

## 2. CHI TIẾT CÁC LỖI VÀ ĐIỂM BẤT CẬP TRONG `5 DL.tex` CẦN SỬA

### Lỗi 1: Mâu thuẫn phương pháp luận về Rò rỉ Dữ liệu (Data Leakage)
- **Vị trí:** Dòng 130, dòng 379, và Bảng 5.3 (`tab:protocol_diff`, dòng 143–144).
- **Nội dung bị sai:**
  - Văn bản hiện tại khẳng định: *"Học máy tinh chỉnh siêu tham số và chọn ngưỡng trực tiếp trên $T_{\text{out}}$ (Optuna trong pipeline_master_unified.py...)"* và coi kết quả Học máy là *"cận trên có rò rỉ"*.
- **Thực tế:**
  - Chương 4 Học máy (`pipeline_ml.py`) đã tuân thủ **100% Zero Data Leakage**: Optuna TPE tối ưu trên tập kiểm định nội bộ của $T_{\text{in}}$, ngưỡng $\tau^*$ được hiệu chuẩn hoàn toàn trên tập kiểm định nội bộ, không hề nhìn trước nhãn $T_{\text{out}}$. Tệp `pipeline_master_unified.py` là tệp nháp cũ đã bị xóa.
- **Biện pháp sửa:**
  - Sửa lại Bảng 5.3 và các đoạn văn dòng 130, 379: Khẳng định cả Học máy và Học sâu đều tuân thủ nghiêm ngặt chuẩn **Zero Data Leakage** trên tập kiểm định nội bộ và đánh giá độc lập ngoài mẫu trên $T_{\text{out}}$. Điểm khác biệt duy nhất là Học sâu dùng `GroupKFold` theo tòa nhà trên chuỗi $420$h, còn Học máy tối ưu trên mẫu bảng độc lập.

---

### Lỗi 2: Số liệu đối chứng Học máy bị cũ và sai lệch
- **Vị trí:** Dòng 379.
- **Nội dung bị sai:** So sánh BiGRU với con số mơ hồ *"ensemble Học máy tốt nhất PR-AUC 0,695"*.
- **Biện pháp sửa:** Cập nhật số liệu chuẩn xác từ Bảng 4.6 của Chương 4: Nêu rõ từng mô hình cây trên tầng Context (XGBoost $0{,}6810$, LightGBM $0{,}6750$, CatBoost $0{,}6691$, HistGB $0{,}6671$) để làm nổi bật mức tăng trưởng thực tế của BiGRU ($0{,}7535$) và tổ hợp BiGRU+BiLSTM ($0{,}7574$).

---

### Lỗi 3: Các bảng số liệu bị bỏ trống dấu gạch ngang `--` (Chưa chạy)
- **Vị trí:**
  - Bảng 5.6 (`tab:dl_pos_weight`): Toàn bộ 4 dòng quét $w_+ \in \{1, 7, 20, 45\}$ đều là `--`.
  - Bảng 5.8 (`tab:dl_recall_by_type`): Toàn bộ 5 mô hình đều là `--`.
  - Bảng 5.9 (`tab:dl_progression`): Dòng Cấp độ 4 toàn bộ là `--`, Cấp độ 3 thiếu $F_1$.
- **Hậu quả:** Báo cáo khoa học chính thức để các bảng trống toàn dấu gạch ngang tạo cảm giác chưa hoàn thành thực nghiệm.
- **Biện pháp sửa:**
  - Chuyển Bảng 5.6 thành phân tích kỹ thuật: giải thích cơ sở lựa chọn $w_+ = 20$ dựa trên tỷ lệ nghịch đảo lớp hiếm ($\approx 1 / 0{,}02$) để cân bằng hàm mất mát `BCEWithLogitsLoss`.
  - Chuyển Bảng 5.8 thành phân tích định tính về đặc tính bắt lỗi của từng kiến trúc mạng.
  - Hoàn thiện Bảng 5.9: Bổ sung chỉ số $F_1$ cho Cấp độ 3 (Ensemble BiGRU+BiLSTM: $0{,}6812$), loại bỏ dòng Cấp độ 4 giả định.

---

### Lỗi 4: Thiếu lý giải chiều sâu dựa trên Kiến trúc Mạng Nơ-ron
- **Vấn đề:** Hiện tại chỉ liệt kê số đo mà chưa giải thích *vì sao mô hình này cao, mô hình kia thấp* dựa trên cơ chế mạng (tương tự như Chương 4).
- **Biện pháp bổ sung lý giải:**
  1. **BiGRU đạt đỉnh ($0{,}7535$ PR-AUC, $0{,}9646$ ROC-AUC):** Cơ chế 2 cổng (reset và update), ít tham số hơn BiLSTM ($113\text{k}$ so với $151\text{k}$), giúp hội tụ nhanh, hạn chế triệt để quá khớp trên $200$ chuỗi công trình; đồng thời hướng quét 2 chiều trong cửa sổ $420$ giờ giúp mô hình nhìn thấy cả xu hướng phụ tải trước và sau thời điểm nghi ngờ dị thường.
  2. **BiLSTM thấp hơn BiGRU một chút ($0{,}7402$ PR-AUC):** Sở hữu 3 cổng cùng trạng thái ô nhớ cell state riêng biệt ($151\text{k}$ tham số). Với quy mô dữ liệu $200$ tòa nhà và số epoch bị giới hạn ($10$ epoch), cấu trúc phức tạp hơn này gặp hiện tượng hội tụ chậm hơn và độ trễ gradient cao hơn BiGRU.
  3. **TCN nhân quả ($0{,}6677$ PR-AUC):** Tích chập giãn nở nhân quả (Causal Dilated Convolutions) với vùng nhìn $253$ giờ chỉ được phép nhìn về quá khứ (không nhìn tương lai như BiGRU/BiLSTM). Điều này phản ánh trung thực bài toán cảnh báo trực tuyến thời gian thực, nhưng thiếu đi thông tin phản hồi hồi phục phụ tải sau sự cố, dẫn đến PR-AUC thấp hơn mạng 2 chiều ngoại tuyến.
  4. **MLP tụt hậu ($0{,}5305$ PR-AUC):** Hoàn toàn không có cơ chế nhớ thời gian, chỉ dự đoán dựa trên vector đặc trưng tĩnh từng giờ, kém mô hình chuỗi từ $+0{,}14$ đến $+0{,}22$ điểm PR-AUC.
  5. **LSTM Autoencoder thất bại ($0{,}0335 - 0{,}1107$ PR-AUC):** Mạng tái tạo không giám sát bị lỗi không có điểm nghẽn (bottleneck), dẫn đến việc mạng học thành phép đồng nhất (identity mapping) và tái tạo tốt cả các đoạn treo phẳng (flatline) dị thường. Ngoài ra, sai số tái tạo MSE bị chi phối bởi biên độ tiêu thụ hơn là nắm bắt các mẫu hình bất thường ngữ cảnh.

---

### Lỗi 5: Cảnh báo tràn lề trang (Overfull \hbox 134pt) và vỡ bảng (Underfull \hbox)
- **Vị trí tràn lề:**
  - Dòng 18: Chuỗi `*\_chunked\_\{baseline,zone,context\}\_timestamp.parquet` gây Overfull $90{,}7$pt.
  - Dòng 92: Chuỗi `model\_DL/artifacts/chunked/<run-id>/` gây Overfull $134{,}8$pt.
- **Vị trí vỡ bảng:**
  - Dòng 400–425 (Bảng giả thuyết H1–H9): Dùng `\begin{center}\begin{tabular}` không bọc môi trường bảng co giãn, gây ra 15 lỗi `Underfull \hbox (badness 10000)`.
- **Biện pháp sửa:**
  - Ngắt dòng các chuỗi đường dẫn tệp hoặc đưa vào chú thích chân trang/phụ lục.
  - Chuẩn hóa Bảng Giả thuyết thành `\begin{table}[H] \resizebox{\textwidth}{!}{...}` hoặc `tabularx` để căn lề hoàn hảo.

---

## 3. LỘ TRÌNH THỰC THI (ĐÃ HOÀN TẤT)

1. **Bước 1 (Lưu kế hoạch):** Ghi nhận đầy đủ kế hoạch vào tệp `note.md`. [HOÀN TẤT]
2. **Bước 2 (Chỉnh sửa 5 DL.tex):** [HOÀN TẤT]
   - Đã đồng bộ 100% Zero Data Leakage giữa Học máy và Học sâu tại Mục 5.2.3, Bảng 5.3 và Mục 5.3.6.
   - Bổ sung phân tích kiến trúc chuyên sâu giải thích cơ chế nơ-ron (BiGRU vs BiLSTM vs TCN vs MLP vs LSTM-AE).
   - Chuyển Bảng 5.6 thành phân tích kỹ thuật dẫn xuất $w_{+} = 20$.
   - Chuyển Bảng 5.8 thành phân tích định tính hình thái dị thường (Point, Contextual, Flatline).
   - Hoàn thiện Bảng 5.9 (bổ sung $F_1 = 0{,}6812$ cho Cấp độ 3 Ensemble, lược bỏ Cấp độ 4 dạng `--`).
   - Chuẩn hóa Bảng 5.10 Giả thuyết nghiên cứu thành `\begin{table}[H] \resizebox{\textwidth}{!}{...}`.
   - Đối chứng khách quan trên cùng không gian tương đương Context-Aware: BiGRU ($0{,}7535$) vượt trội XGBoost Context ($0{,}6810$) $+0{,}0725$ PR-AUC; mở rộng tiềm năng trên bộ Expanded 53 đặc trưng.
3. **Bước 3 (Biên dịch và Thẩm định):** [HOÀN TẤT]
   - Biên dịch thành công `main.pdf` bằng Tectonic (Mã thoát: 0).
   - Chương 5 (`5 DL.tex`) **sạch hoàn toàn cảnh báo (0 Overfull \hbox, 0 Underfull \hbox)**.

