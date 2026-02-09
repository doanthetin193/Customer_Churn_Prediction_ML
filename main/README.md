# ML Churn Prediction - Project Documentation

## 📋 Tổng quan

Project thực hiện **dự đoán khách hàng rời bỏ (Customer Churn)** trong ngành viễn thông, dựa trên bài báo nghiên cứu. Sử dụng 5 thuật toán Machine Learning và kỹ thuật XAI (LIME, SHAP) để giải thích mô hình.

---

## 📁 Cấu trúc Project

```
d:\ML\main\
├── requirements.txt      # Thư viện cần cài
├── main.py               # Entry point - chạy toàn bộ pipeline
├── data_loader.py        # Load dataset
├── data_preprocessing.py # Xử lý dữ liệu
├── models.py             # 5 thuật toán ML
├── evaluation.py         # Đánh giá & vẽ biểu đồ
├── explainability.py     # LIME & SHAP
└── results/              # Thư mục chứa kết quả
```

---

## 📄 Chi tiết từng file

### 1. `requirements.txt`
Danh sách thư viện Python cần cài đặt:
- `pandas`, `numpy` - Xử lý dữ liệu
- `scikit-learn` - Thuật toán ML
- `matplotlib`, `seaborn` - Vẽ biểu đồ
- `lime`, `shap` - Giải thích mô hình
- `imbalanced-learn` - SMOTE (optional)

### 2. `data_loader.py`
**Chức năng:** Load dataset và thống kê cơ bản
- `load_data()` - Đọc file CSV
- `explore_data()` - In thông tin: số dòng, cột, % churn

### 3. `data_preprocessing.py`
**Chức năng:** Xử lý dữ liệu trước khi train
- `create_target_variable()` - Tạo biến target (Churn = 0/1)
- `select_features()` - Chọn features theo paper
- `handle_missing_values()` - Điền giá trị thiếu
- `encode_categorical()` - Encode Yes/No → 0/1
- `scale_features()` - Chuẩn hóa dữ liệu (StandardScaler)
- `preprocess_data()` - Pipeline tổng hợp + SMOTE (optional)

### 4. `models.py`
**Chức năng:** Định nghĩa và train 5 thuật toán ML

| # | Thuật toán | Thư viện |
|---|------------|----------|
| 1 | Logistic Regression | `sklearn.linear_model` |
| 2 | KNN | `sklearn.neighbors` |
| 3 | Naive Bayes | `sklearn.naive_bayes` |
| 4 | Decision Tree | `sklearn.tree` |
| 5 | Random Forest | `sklearn.ensemble` |

- `get_models()` - Trả về dict 5 models với hyperparameters đã tối ưu
- `get_tuned_random_forest()` - GridSearchCV cho Random Forest
- `train_all_models()` - Train tất cả models
- `predict_all_models()` - Dự đoán trên test set

### 5. `evaluation.py`
**Chức năng:** Tính metrics và vẽ biểu đồ
- `calculate_metrics()` - Accuracy, Sensitivity, Specificity, AUC
- `plot_confusion_matrices()` - Figure 5
- `plot_roc_curves()` - Figure 6
- `plot_metrics_comparison()` - Figure 7

### 6. `explainability.py`
**Chức năng:** Giải thích mô hình với XAI
- `create_lime_explainer()` - Tạo LIME explainer
- `explain_with_lime()` - Giải thích 1 sample (Figure 8)
- `explain_with_shap()` - SHAP summary plot (Figure 9, 10)

### 7. `main.py`
**Entry point** - Chạy toàn bộ pipeline:
1. Load data
2. Preprocess
3. Train 5 models
4. Evaluate
5. Generate plots
6. XAI explanations

---

## 🚀 Cách chạy

```bash
# Cài thư viện
pip install -r requirements.txt

# Chạy thực nghiệm
cd d:\ML\main
python main.py
```

---

## 📊 Kết quả sinh ra (trong `results/`)

| File | Mô tả |
|------|-------|
| `confusion_matrices.png` | Ma trận nhầm lẫn 5 models |
| `roc_curves.png` | Đường cong ROC |
| `metrics_comparison.png` | So sánh Accuracy, AUC, Sensitivity, Specificity |
| `lime_sample_0.png` | LIME giải thích sample 1 |
| `lime_sample_1.png` | LIME giải thích sample 2 |
| `shap_summary.png` | SHAP feature importance |
| `shap_bar.png` | SHAP bar plot |

---

## 📈 Kết quả thực nghiệm

### So sánh tất cả Models (với SMOTE):

| Model | Accuracy | AUC | Sensitivity | Specificity |
|-------|:--------:|:---:|:-----------:|:-----------:|
| **Random Forest** | **82%** | **0.8892** | **72%** | **86%** |
| Logistic Regression | 77% | 0.8837 | 84% | 74% |
| Decision Tree | 80% | 0.7918 | 69% | 84% |
| Naive Bayes | 76% | 0.8478 | 79% | 75% |
| KNN | 73% | 0.8271 | 80% | 70% |

### 🏆 Random Forest (Best Model) - So sánh SMOTE:

| Metric | Không SMOTE | Có SMOTE | Paper | Ghi chú |
|--------|:-----------:|:--------:|:-----:|---------|
| **AUC** | 0.8958 | 0.8892 | 0.95 | Giảm nhẹ |
| **Accuracy** | 83% | 82% | 87% | Giảm nhẹ |
| **Sensitivity** | 62% | **72%** | 85% | **↑10%** ✅ |
| **Specificity** | 91% | 86% | 88% | Trade-off |

### Thứ tự AUC:
Random Forest > Logistic Regression > Naive Bayes > KNN > Decision Tree

---

## ⚙️ Cấu hình hiện tại

### SMOTE (đang BẬT - khuyến nghị):
Trong `main.py`, dòng 38:
```python
data = preprocess_data(df, use_smote=True)  # Enable SMOTE for better Sensitivity
```

**Lợi ích SMOTE:**
- Cân bằng dữ liệu: Churn rate từ 26.5% → 50%
- Train size tăng: 5,282 → 7,760 samples
- Sensitivity tăng: 62% → 72% (+10%)
- Trade-off: Specificity giảm 5% (chấp nhận được)

### Tắt SMOTE (nếu muốn):
```python
data = preprocess_data(df, use_smote=False)
```

### Tắt GridSearchCV (chạy nhanh hơn):
```python
trained_models = train_all_models(data['X_train'], data['y_train'], tune_rf=False)
```

---

## 📝 Ghi chú

- **Dataset**: `telecom_customer_churn.csv` (7,043 customers, 38 features → 25 features sau khi chọn lọc)
- **Train/Test split**: 75/25 (5,282 / 1,761 samples)
- **Random state**: 42 (để reproducible)
- **Reference**: "Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models" - Algorithms 2024
