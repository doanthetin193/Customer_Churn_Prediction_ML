# Tài liệu Pipeline Machine Learning

**Học phần:** Học Máy và Ứng Dụng  
**Đề tài:** Dự đoán rời bỏ khách hàng trong ngành viễn thông  
**Tham chiếu bài báo:** Chang et al. (2024). *Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models.* Algorithms, 17(6), 231. https://doi.org/10.3390/a17060231

---

## 1. Tổng quan pipeline

Pipeline ML nằm trong thư mục `main/` và được điều phối bởi `main/main.py`. Gồm 7 bước tuần tự:

```
Load Data → Preprocess → Train (5 models) → Predict → Evaluate → Plot → XAI (LIME + SHAP)
```

---

## 2. Dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | Maven Analytics — Telecom Customer Churn |
| File | `dataset/telecom_customer_churn.csv` |
| Kích thước gốc | 7.043 dòng, 38 cột |
| Kích thước sau lọc | **6.589 dòng** (loại bỏ nhóm "Joined") |
| Tỉ lệ churn (sau lọc) | ~28.4% (Churned) / ~71.6% (Stayed) |

### Lý do loại bỏ khách hàng "Joined"

Dataset gốc có 3 nhãn: `Churned`, `Stayed`, `Joined` (454 khách hàng mới gia nhập).  
Nhóm `Joined` **chưa có cơ hội rời bỏ** nên không phải đối tượng dự đoán churn.  
Bài báo gốc sử dụng ~4.601 mẫu (với Internet service), xác nhận việc không dùng toàn bộ 7.043 mẫu.  
Sau khi loại `Joined`, còn **6.589 mẫu** — sát với tinh thần bài báo hơn.

---

## 3. Tiền xử lý dữ liệu (`data_preprocessing.py`)

### 3.1. Biến mục tiêu

```python
Churn = 1  nếu Customer Status == "Churned"
Churn = 0  nếu Customer Status == "Stayed"
```

### 3.2. Chọn features

25 features được chọn theo bài báo:

| Nhóm | Features |
|---|---|
| Demographics | Gender, Age, Married, Number of Dependents |
| Hành vi | Number of Referrals, Tenure in Months, Offer |
| Dịch vụ | Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| Hợp đồng / Thanh toán | Contract, Paperless Billing, Payment Method |
| Tài chính | Monthly Charge, Total Charges, Total Revenue |

### 3.3. Xử lý missing values

- Cột số (`numeric`): điền bằng **median**
- Cột phân loại (`categorical`): điền bằng **mode**

### 3.4. Mã hóa và chuẩn hóa

- **LabelEncoder** cho 18 cột phân loại
- **StandardScaler** cho toàn bộ features (fit trên train, transform trên test)

### 3.5. Chia tập dữ liệu

```
Train : Test = 75% : 25%   (stratify=True, random_state=42)
Train = 4.941 mẫu
Test  = 1.648 mẫu
```

Tỉ lệ 75/25 theo đúng bài báo: *"the research data was divided into Train (75%) and Test (25%)"*.

### 3.6. Xử lý mất cân bằng dữ liệu

**Không dùng SMOTE** — bài báo giữ nguyên tỉ lệ tự nhiên.  
Thay vào đó dùng **`class_weight='balanced'`** trong sklearn cho các mô hình LR, Decision Tree, Random Forest — đây là cách xử lý chuẩn mà không tạo dữ liệu giả.

> **So sánh:** Khi bật SMOTE, Sensitivity tăng giả tạo (LR: 83.94%) nhưng Accuracy và AUC của RF giảm (82.28% / 0.8892). Sau khi tắt SMOTE và bật `class_weight='balanced'`, RF đạt Accuracy **86.65%** và AUC **0.9264**.

---

## 4. Các mô hình ML (`models.py`)

### 4.1. Danh sách 5 thuật toán (theo bài báo)

| Model | Cấu hình |
|---|---|
| **Logistic Regression** | `max_iter=1000`, `class_weight='balanced'`, `random_state=42` |
| **KNN** | `n_neighbors=5`, `weights='distance'` |
| **Naïve Bayes** | Mặc định `GaussianNB()` |
| **Decision Tree** | `class_weight='balanced'`, `random_state=42` |
| **Random Forest** | `n_estimators=200`, `max_depth=15`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `class_weight='balanced'`, `random_state=42` |

### 4.2. Lưu ý về hyperparameters

- Bài báo không công bố hyperparameters cụ thể. Các giá trị trên là **reasonable defaults** dựa trên best practice.
- **Không dùng GridSearchCV** (bài báo không đề cập) → `tune_rf=False` trong `main.py`.
- `class_weight='balanced'` được thêm vào LR / DT / RF để bù đắp mất cân bằng lớp mà không cần SMOTE.

---

## 5. Đánh giá mô hình (`evaluation.py`)

### 5.1. Các chỉ số theo bài báo

| Chỉ số | Công thức |
|---|---|
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| Sensitivity (Recall/TPR) | TP / (TP + FN) |
| Specificity (TNR) | TN / (TN + FP) |
| AUC-ROC | Diện tích dưới đường ROC |

### 5.2. Kết quả thực nghiệm

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.46% | 86.72% | 77.98% | 0.9112 |
| KNN | 81.01% | 66.81% | 86.62% | 0.8521 |
| Naïve Bayes | 80.10% | 78.59% | 80.69% | 0.8771 |
| Decision Tree | 82.10% | 68.09% | 87.64% | 0.7787 |
| **Random Forest** | **86.65%** | **74.52%** | **91.45%** | **0.9264** |

**Best model: Random Forest** (AUC = 0.9264)

### 5.3. So sánh với bài báo gốc

| Model | Bài báo — Accuracy | Bài báo — AUC | Project — Accuracy | Project — AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 75.53% | 0.84 | 80.46% | 0.9112 |
| KNN | 71.29% | 0.81 | 81.01% | 0.8521 |
| Naïve Bayes | 79.84% | 0.88 | 80.10% | 0.8771 |
| Decision Tree | 80.04% | 0.80 | 82.10% | 0.7787 |
| **Random Forest** | **86.94%** | **0.95** | **86.65%** | **0.9264** |

**Nhận xét:**
- Thứ tự ranking các mô hình **hoàn toàn khớp** bài báo (RF > LR ≈ NB > DT > KNN theo AUC).
- Accuracy RF chênh lệch chỉ **0.29%** (86.65% vs 86.94%).
- Sai khác còn lại là do bài báo không công khai bước lọc subset (paper dùng 4.601 mẫu, project dùng 6.589 mẫu sau lọc Joined).

---

## 6. Đầu ra hình ảnh (`main/results/`)

| File | Mô tả | Tương ứng bài báo |
|---|---|---|
| `confusion_matrices.png` | Confusion matrix của 5 mô hình | Figure 5 |
| `roc_curves.png` | ROC curve của 5 mô hình | Figure 6 |
| `metrics_comparison.png` | Biểu đồ so sánh Accuracy / Sensitivity / Specificity / AUC | Figure 7 |
| `lime_sample_0.png` | Giải thích LIME cho mẫu 0 (class "Good") | Figure 8 |
| `lime_sample_1.png` | Giải thích LIME cho mẫu 1 | Figure 8 |
| `shap_summary.png` | SHAP summary plot (beeswarm) | Figure 9 |
| `shap_bar.png` | SHAP bar plot (feature importance tổng thể) | Figure 10 |

---

## 7. Giải thích mô hình XAI (`explainability.py`)

Áp dụng cho **Random Forest** (best model).

### 7.1. LIME

- Giải thích **cục bộ** (từng mẫu) — tại sao mô hình dự đoán nhãn X cho khách hàng cụ thể.
- `class_names = ['Good', 'Bad']` theo bài báo (Figure 8).
- Giải thích class 0 ("Good" / Not Churned) để khớp Figure 8.
- 2 mẫu được giải thích: `lime_sample_0.png`, `lime_sample_1.png`.

### 7.2. SHAP

- Giải thích **toàn cục** — tầm quan trọng của từng feature trên toàn bộ tập test.
- Dùng 100 mẫu test đầu tiên để tính toán nhanh.
- `shap_summary.png`: beeswarm plot, màu đỏ = giá trị feature cao, màu xanh = thấp.
- `shap_bar.png`: mean |SHAP value| theo feature — dễ đọc cho báo cáo.

---

## 8. Cách chạy

```bash
# Từ thư mục gốc của project
python main/main.py
```

Hoặc với virtual environment:

```bash
d:\Customer_Churn_Prediction_ML\.venv\Scripts\python.exe main/main.py
```

---

## 9. Cấu trúc thư mục `main/`

```
main/
├── main.py               # Entry point, điều phối toàn bộ pipeline
├── data_loader.py        # Load CSV, thống kê phân phối nhãn
├── data_preprocessing.py # Filter, encode, scale, split
├── models.py             # Định nghĩa và train 5 mô hình
├── evaluation.py         # Tính metrics, vẽ biểu đồ
├── explainability.py     # LIME và SHAP
└── results/              # Đầu ra hình ảnh (tự sinh khi chạy)
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── metrics_comparison.png
    ├── lime_sample_0.png
    ├── lime_sample_1.png
    ├── shap_summary.png
    └── shap_bar.png
```

---

## 10. Các thay đổi quan trọng so với phiên bản ban đầu

| # | Thay đổi | Trước | Sau | Tác động |
|---|---|---|---|---|
| 1 | Filter "Joined" customers | Không lọc (7.043 mẫu) | Lọc còn 6.589 mẫu | Sát bài báo hơn |
| 2 | SMOTE | Bật (`use_smote=True`) | Tắt (`use_smote=False`) | Không tạo dữ liệu giả |
| 3 | Class imbalance | SMOTE | `class_weight='balanced'` | Chuẩn sklearn, không bias |
| 4 | GridSearchCV RF | Bật (`tune_rf=True`) | Tắt (`tune_rf=False`) | Đúng theo bài báo |
| 5 | RF hyperparams | n_estimators=100 (default) | n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2 | AUC RF: 0.8892 → **0.9264** |
| 6 | KNN | n_neighbors=7, weights=distance | n_neighbors=5, weights=distance | Gần default hơn |
