# Giải thích: `main.py` — Điểm Khởi Động Pipeline

## Vị trí trong Pipeline

```
[ENTRY POINT] main.py  ← Bạn đang ở đây
 ├─► BƯỚC 1: data_loader.py
 ├─► BƯỚC 2: data_preprocessing.py
 ├─► BƯỚC 3: models.py
 ├─► BƯỚC 4: evaluation.py
 └─► BƯỚC 5: explainability.py
```

`main.py` là file **điều phối (orchestrator)** — không chứa logic phức tạp, chỉ gọi các module theo đúng thứ tự.

---

## Mục đích

Chạy toàn bộ pipeline ML end-to-end theo phương pháp của bài báo Chang et al. (2024):

> Tải dữ liệu → Tiền xử lý → Huấn luyện 5 model → Dự đoán → Đánh giá → Vẽ biểu đồ → Giải thích XAI

---

## Liên hệ với Bài Báo (Section 3 — Methodology)

| Bước trong main.py | Section bài báo | Tương ứng với |
|---|---|---|
| BƯỚC 1: `load_data()` | Section 3.2 | Maven Analytics dataset, 7043 rows |
| BƯỚC 2: `preprocess_data(use_smote=False)` | Section 3.2 | 75/25 split, StandardScaler, loại "Joined" |
| BƯỚC 3: `train_all_models(tune_rf=False)` | Section 3.3 | 5 algorithms: LR, KNN, NB, DT, RF |
| BƯỚC 4: `predict_all_models()` | Section 3.3 | Inference trên test set |
| BƯỚC 5: `evaluate_all_models()` | Section 3.4 | Accuracy, Sensitivity, Specificity, AUC |
| BƯỚC 6: `generate_all_plots()` | Figures 5,6,7 | Confusion Matrix, ROC, Comparison |
| BƯỚC 7: `generate_explanations()` | Section 3.5 | LIME (Fig 8), SHAP (Fig 9, 10) |

---

## Phân tích Code Chi Tiết

### Header và Import

```python
"""
Main Entry Point
Churn Prediction ML Experiment - Complete Pipeline

Based on: "Predicting Customer Churn in the Telecommunications Industry"
Implements: 5 ML Algorithms + Evaluation + XAI (LIME/SHAP)
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, explore_data
from data_preprocessing import preprocess_data
from models import train_all_models, predict_all_models
from evaluation import evaluate_all_models, print_results_table, generate_all_plots
from explainability import generate_explanations
```

**`sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))`:**

| Phần | Ý nghĩa |
|---|---|
| `__file__` | Đường dẫn tuyệt đối của `main.py` |
| `os.path.abspath(__file__)` | Resolve symlink → absolute path thực sự |
| `os.path.dirname(...)` | Lấy thư mục chứa file = `main/` |
| `sys.path.insert(0, ...)` | Thêm `main/` vào đầu Python import path |

**Tại sao cần điều này?**

Python tìm module theo `sys.path`. Nếu chạy từ project root:

```bash
cd D:\Customer_Churn_Prediction_ML
python main/main.py
```

`from data_loader import ...` sẽ tìm `data_loader.py` trong thư mục hiện tại (project root) → **không tìm thấy** vì file nằm trong `main/`.

`sys.path.insert(0, ...)` thêm `main/` vào path → Python tìm thấy `data_loader.py` trong `main/`.

**`insert(0, ...)` thay vì `append(...)`:** Thêm vào đầu danh sách → ưu tiên cao nhất, tránh bị shadowed bởi module cùng tên ở nơi khác.

---

### Hàm `main()` — 7 bước pipeline

#### Bước 1: Tải dữ liệu

```python
print("\n📁 STEP 1: LOADING DATA")
df = load_data()
explore_data(df)
```

`load_data()` → DataFrame (7043×38, chưa xử lý)
`explore_data(df)` → In thống kê, không biến đổi data

---

#### Bước 2: Tiền xử lý

```python
print("\n🔧 STEP 2: PREPROCESSING DATA")
data = preprocess_data(df, use_smote=False)  # Paper does NOT use SMOTE
```

**`use_smote=False` — quyết định quan trọng nhất:**

| Phương án | Kỹ thuật | Kết quả RF Accuracy |
|---|---|---|
| Trước (cũ) | `use_smote=True` | 82.28% |
| Hiện tại | `use_smote=False` + `class_weight='balanced'` | **86.65%** |
| Bài báo | Không dùng SMOTE | **86.94%** |

Bài báo không đề cập SMOTE trong Methodology. Thay vào đó, điều chỉnh class imbalance bằng cách tăng `class_weight` trong model → gần đúng hơn với cách bài báo tiếp cận.

**`data` là dict với keys:**

```python
data = {
    'X_train': ndarray(4941, 25),    # Scaled features train
    'X_test':  ndarray(1648, 25),    # Scaled features test
    'y_train': ndarray(4941,),       # Labels train (0/1)
    'y_test':  ndarray(1648,),       # Labels test (0/1)
    'feature_names': list(25),       # Tên 25 feature
    'encoders': dict,                # LabelEncoder instances
    'scaler': StandardScaler,        # Để decode sau này
    'X_train_df': DataFrame,         # Dạng DataFrame cho LIME
    'X_test_df': DataFrame           # Dạng DataFrame cho LIME
}
```

---

#### Bước 3: Huấn luyện model

```python
print("\n🤖 STEP 3: TRAINING MODELS")
trained_models = train_all_models(
    data['X_train'], data['y_train'],
    tune_rf=False  # Paper does not mention GridSearchCV
)
```

**`tune_rf=False` — không dùng GridSearchCV:**

Bài báo không đề cập đến hyperparameter tuning bằng GridSearch. Các hyperparameter đã được hardcode trực tiếp trong `get_models()` theo kiến thức domain và best practices cho bài toán churn. Điều này khớp với methodology của bài báo tốt hơn.

`trained_models` là dict: `{'Logistic Regression': model, 'KNN': model, ...}`

---

#### Bước 4: Dự đoán

```python
print("\n📊 STEP 4: MAKING PREDICTIONS")
predictions = predict_all_models(trained_models, data['X_test'])
```

`predictions` là dict:

```python
{
    'Logistic Regression': {'y_pred': array([0,1,0,...]), 'y_prob': array([0.23, 0.87,...])},
    'KNN':                 {'y_pred': ..., 'y_prob': ...},
    ...
}
```

`y_pred` dùng threshold mặc định 0.5 (khác với DL pipeline — ở ML pipeline ta dùng threshold cố định).

---

#### Bước 5: Đánh giá

```python
print("\n📈 STEP 5: EVALUATING MODELS")
results = evaluate_all_models(data['y_test'], predictions)
print_results_table(results)
```

**Kết quả thực tế sau khi tối ưu:**

```
==================================================================
MODEL COMPARISON TABLE
==================================================================
Model                  Accuracy  Sensitivity  Specificity      AUC
------------------------------------------------------------------
Logistic Regression      80.46%       86.72%       77.98%   0.9112
KNN                      81.01%       66.81%       86.62%   0.8521
Naive Bayes              80.10%       78.59%       80.69%   0.8771
Decision Tree            82.10%       68.09%       87.64%   0.7787
Random Forest            86.65%       74.52%       91.45%   0.9264
------------------------------------------------------------------
🏆 Best Model: Random Forest (AUC = 0.9264)
```

---

#### Bước 6: Vẽ biểu đồ

```python
print("\n🎨 STEP 6: GENERATING PLOTS")
generate_all_plots(data['y_test'], predictions, results, results_dir)
```

Lưu 3 file vào `main/results/`:
- `confusion_matrices.png` (Figure 5)
- `roc_curves.png` (Figure 6)
- `metrics_comparison.png` (Figure 7)

---

#### Bước 7: XAI Explanations

```python
print("\n🔍 STEP 7: XAI EXPLANATIONS")
rf_model = trained_models['Random Forest']
generate_explanations(
    rf_model,
    data['X_train'],
    data['X_test'],
    data['feature_names'],
    results_dir
)
```

**Tại sao chỉ giải thích Random Forest?**

> Bài báo Section 3.5: *"The RF algorithm was selected to be explained using the XAI methods since it presented the best results among all models."*

Random Forest đạt AUC cao nhất (0.9264) → được chọn làm model đại diện để giải thích. Giải thích model tốt nhất có giá trị thực tế nhất: nếu deploy model này, ta cần biết nó dựa vào đâu để quyết định.

Lưu 4 file:
- `lime_sample_0.png` (Figure 8)
- `lime_sample_1.png`
- `shap_summary.png` (Figure 9)
- `shap_bar.png` (Figure 10)

---

### Cấu trúc `main()` — if __name__ == "__main__"

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

**`if __name__ == "__main__"`:**

Khi Python chạy file trực tiếp (`python main.py`), `__name__` = `"__main__"` → block này chạy.
Khi file được `import` (ví dụ: `from main import something`), `__name__` = `"main"` → block này KHÔNG chạy.

Đây là Python idiom chuẩn để phân biệt "chạy trực tiếp" và "import as module".

---

### Setup `results_dir`

```python
results_dir = os.path.join(os.path.dirname(__file__), 'results')
```

Tạo đường dẫn tuyệt đối đến `main/results/`:

```
os.path.dirname(__file__) → D:\Customer_Churn_Prediction_ML\main
os.path.join(..., 'results') → D:\Customer_Churn_Prediction_ML\main\results
```

Các hàm `plot_*` và `generate_explanations` sẽ tự tạo thư mục `results/` nếu chưa tồn tại qua `ensure_results_dir()`.

---

## Cách chạy

```bash
# Từ project root
cd D:\Customer_Churn_Prediction_ML
python main/main.py

# Hoặc từ thư mục main
cd D:\Customer_Churn_Prediction_ML\main
python main.py
```

**Thời gian chạy ước tính:**

| Bước | Thời gian |
|---|---|
| Bước 1-2: Load + Preprocess | ~2 giây |
| Bước 3: Train 5 models | ~15-30 giây |
| Bước 4-5: Predict + Evaluate | ~1 giây |
| Bước 6: Generate 3 plots | ~5 giây |
| Bước 7: LIME (2 samples) | ~10 giây |
| Bước 7: SHAP (100 samples, RF) | ~30-60 giây |
| **Tổng** | **~1-2 phút** |

---

## Sơ đồ luồng dữ liệu tổng thể

```
telecom_customer_churn.csv
        │
        ▼
 load_data()
        │ DataFrame(7043×38)
        ▼
 preprocess_data(use_smote=False)
        │ X_train(4941×25)  X_test(1648×25)
        │ y_train(4941,)    y_test(1648,)
        ▼
 train_all_models(tune_rf=False)
        │ {model_name: fitted_model}
        ▼
 predict_all_models()
        │ {model_name: {y_pred, y_prob}}
        ├──────────────────────────────┐
        ▼                              ▼
 evaluate_all_models()         generate_explanations(RF)
 print_results_table()              │
 generate_all_plots()               ├─ lime_sample_0.png
        │                           ├─ lime_sample_1.png
        ├─ confusion_matrices.png   ├─ shap_summary.png
        ├─ roc_curves.png           └─ shap_bar.png
        └─ metrics_comparison.png
```

---

## Tóm tắt các thay đổi so với bài báo

| Quyết định | Bài báo | Code hiện tại | Lý do |
|---|---|---|---|
| SMOTE | Không đề cập | `use_smote=False` | Khớp bài báo |
| Class imbalance | Không đề cập rõ | `class_weight='balanced'` | Thay thế SMOTE |
| GridSearch | Không đề cập | `tune_rf=False` | Khớp bài báo, đủ nhanh |
| "Joined" customers | Không dùng | Lọc ra trong `preprocess_data` | Dataset khớp intent bài báo |
| RF n_estimators | Không nêu cụ thể | 200 | Best practice cho RF |
| Threshold | 0.5 | 0.5 (ML) | ML dùng default |
