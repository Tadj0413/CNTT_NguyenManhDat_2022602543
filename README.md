# 💳 Hệ Thống Phát Hiện Gian Lận Thẻ Tín Dụng (Credit Card Fraud Detection System)

> **Đề tài tốt nghiệp ngành Công nghệ thông tin**  
> **Sinh viên thực hiện:** Nguyễn Mạnh Đạt  
> **Mã số sinh viên:** 2022602543  
> **Trường:** [Tên trường Đại học của bạn] (ví dụ: Đại học Công nghiệp Hà Nội)

Hệ thống ứng dụng các mô hình học máy hiện đại (Machine Learning) kết hợp với các kỹ thuật xử lý mất cân bằng dữ liệu để phát hiện và cảnh báo các giao dịch gian lận thẻ tín dụng theo thời gian thực với độ chính xác cao.

---

## 🌟 Tính Năng Nổi Bật

- 🤖 **Đa Dạng Thuật Toán Học Máy:** Tích hợp 3 mô hình học máy phổ biến và mạnh mẽ nhất hiện nay cho bài toán phân loại:
  - **Random Forest Classifier**
  - **XGBoost Classifier**
  - **LightGBM Classifier**
- ⚖️ **Xử Lý Mất Cân Bằng Dữ Liệu:** Áp dụng các phương pháp lấy mẫu nâng cao như **SMOTE (Synthetic Minority Over-sampling Technique)** và **ADASYN (Adaptive Synthetic)** nhằm nâng cao khả năng bắt lỗi giao dịch gian lận (Recall).
- 📊 **Dashboard Trực Quan (Streamlit UI):**
  - Giao diện thân thiện, hiện đại, dễ thao tác.
  - Tải file CSV giao dịch để kiểm tra hàng loạt.
  - Tùy chỉnh ngưỡng phân loại (Classification Threshold) động với thanh trượt trực quan.
  - Hiển thị kết quả chi tiết từng giao dịch kèm các chỉ số đánh giá (TP, FP, TN, FN).
- 🔍 **Giải Thích Mô Hình (Explainable AI - XAI):** Trực quan hóa mức đóng góp của từng đặc trưng (Feature Contributions) đối với từng giao dịch cụ thể (Top 8 đặc trưng quan trọng nhất) bằng biểu đồ động Plotly.

---

## 🛠️ Công Nghệ Sử Dụng

- **Ngôn ngữ:** Python 3.10+
- **Giao diện người dùng:** Streamlit
- **Xử lý & Phân tích dữ liệu:** Pandas, NumPy, SciPy
- **Học máy & Xử lý mất cân bằng:** Scikit-Learn, Imbalanced-Learn, XGBoost, LightGBM
- **Trực quan hóa:** Plotly, Matplotlib, Seaborn
- **Lưu trữ mô hình:** Joblib

---

## 📂 Cấu Trúc Thư Mục Dự Án

```text
DATN2026/
├── models/                           # Thư mục lưu trữ các mô hình đã huấn luyện (.pkl)
│   ├── model_random forest_smote.pkl
│   ├── model_random forest_adasyn.pkl
│   ├── model_xgboost_smote.pkl
│   ├── model_xgboost_adasyn.pkl
│   ├── model_lightgbm_smote.pkl
│   ├── model_lightgbm_adasyn.pkl
│   └── performance_metrics.csv       # Bảng kết quả đánh giá mô hình
├── data/                             # Thư mục chứa tập dữ liệu mẫu (đã được cấu hình .gitignore)
├── plot/                             # Các biểu đồ được sinh ra trong quá trình huấn luyện/EDA
├── eda_des_analys.ipynb              # Notebook phân tích khám phá dữ liệu (EDA)
├── preprocess.ipynb                  # Notebook tiền xử lý dữ liệu
├── train_origin_data.ipynb           # Notebook huấn luyện mô hình trên dữ liệu gốc
├── fraud_detection_system.ipynb      # Notebook chính huấn luyện, đánh giá & so sánh các mô hình
├── app.py                            # 🚀 Tệp chạy giao diện Web Dashboard (Streamlit)
├── requirements.txt                  # Danh sách thư viện phụ thuộc
├── .gitignore                        # Cấu hình bỏ qua các tệp tin không cần thiết khi push Git
└── README.md                         # Tài liệu hướng dẫn sử dụng hệ thống
```

---

## 📈 Kết Quả Đánh Giá Mô Hình

Dưới đây là bảng so sánh hiệu suất thực tế của các mô hình khi kiểm thử trên tập dữ liệu thử nghiệm (test set):

| Thuật Toán | Phương Pháp | Thời Gian Huấn Luyện (s) | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | SMOTE | 117.43 | **91.25%** | 76.84% | **83.43%** | 96.79% | **80.48%** |
| **Random Forest** | ADASYN | 108.56 | 90.12% | 76.84% | 82.95% | 96.19% | 80.23% |
| **XGBoost** | SMOTE | 7.35 | 62.50% | 78.95% | 69.77% | **97.09%** | 79.96% |
| **XGBoost** | ADASYN | 6.27 | 50.00% | **80.00%** | 61.54% | 96.67% | 78.22% |
| **LightGBM** | SMOTE | 4.03 | 64.96% | **80.00%** | 71.70% | 96.75% | 80.24% |
| **LightGBM** | ADASYN | **3.81** | 50.33% | **80.00%** | 61.79% | 96.11% | 77.96% |

> 📌 **Nhận xét:**
> - **Random Forest** kết hợp với **SMOTE/ADASYN** cho độ chính xác cao nhất (Precision > 90%), hạn chế tối đa việc cảnh báo nhầm (False Positive).
> - **LightGBM** và **XGBoost** có tốc độ huấn luyện cực kỳ nhanh và đạt chỉ số Recall cao nhất (80.00%), giúp giảm thiểu việc bỏ lọt các giao dịch gian lận thực tế (False Negative).

---

## 🚀 Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Yêu Cầu Hệ Thống
* Hệ điều hành: Windows / macOS / Linux
* Phiên bản Python: **Python 3.10** hoặc mới hơn.

### 2. Cài Đặt Môi Trường
1. **Clone repository này về máy:**
   ```bash
   git clone https://github.com/Tadj0413/CNTT_NguyenManhDat_2022602543.git
   cd CNTT_NguyenManhDat_2022602543
   ```

2. **Khởi tạo môi trường ảo (venv):**
   ```bash
   python -m venv .venv
   ```

3. **Kích hoạt môi trường ảo:**
   * **Trên Windows:**
     ```bash
     .venv\Scripts\activate
     ```
   * **Trên macOS/Linux:**
     ```bash
     source .venv/bin/activate
     ```

4. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

### 3. Khởi Chạy Ứng Dụng Streamlit
Chạy lệnh sau tại thư mục gốc của dự án để khởi động Dashboard:
```bash
streamlit run app.py
```

Sau khi chạy lệnh, Streamlit sẽ tự động mở trình duyệt mặc định với địa chỉ mặc định là: `http://localhost:8501`.

### 4. Cách Sử Dụng Dashboard
- **Bước 1:** Chuyển qua Tab **"⚙️ Chọn Mô Hình"** để chọn mô hình và phương pháp lấy mẫu mà bạn muốn sử dụng (ví dụ: Random Forest + SMOTE).
- **Bước 2:** Chuyển sang Tab **"📊 Phân Tích CSV"**, tải lên tệp CSV giao dịch mẫu cần phân tích (yêu cầu tệp có các cột đặc trưng từ `V1` đến `V28`, cột `Time` và `Amount`).
- **Bước 3:** Nhấp nút **⚡ CHẠY PHÂN TÍCH**.
- **Bước 4:** Sử dụng thanh trượt thay đổi ngưỡng (Threshold) phân loại và xem kết quả phân tích trực quan hóa, bao gồm cả đóng góp của từng đặc trưng đối với các giao dịch đáng ngờ!

---

## 🎓 Giáo Viên Hướng Dẫn
* **Thầy/Cô:** [Tên Giáo Viên Hướng Dẫn]
* **Khoa:** Công Nghệ Thông Tin
