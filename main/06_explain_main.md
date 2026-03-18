# Giải thích: `main.py` — Điều Phối Toàn Bộ Pipeline

## Vị trí trong Pipeline

```
[BƯỚC 0] main.py  ← Bạn đang ở đây
  │
  ├─► [1] data_loader.py       → load_data(), explore_data()
  ├─► [2] data_preprocessing.py → preprocess_data()
  ├─► [3] models.py            → train_all_models(), predict_all_models()
  ├─► [4] evaluation.py        → evaluate_all_models(), print_results_table()
  ├─► [5] evaluation.py        → generate_all_plots()
  └─► [6] explainability.py    → generate_explanations()
```

---

## Tổng Quan

`main.py` là **entry point** của toàn bộ project — file duy nhất người dùng cần chạy. Nó không tự chứa logic ML nào; thay vào đó nó **nhập và gọi** các hàm từ 4 module chuyên biệt theo đúng thứ tự.

Đây là pattern kiến trúc phổ biến trong ML projects: **mỗi file làm đúng 1 nhiệm vụ**, `main.py` chịu trách nhiệm kết nối chúng.

---

## Liên Hệ Bài Báo

`main.py` phản ánh **toàn bộ Section 3 (Methodology)** của bài báo Chang et al. (2024):

| Phần trong bài báo | Bước trong main.py |
|---|---|
| Section 3.2 — Dataset | Step 1: Load Data |
| Section 3.2 — Data Split 75/25 | Step 2: Preprocess |
| Section 3.3 — Five ML Algorithms | Step 3: Train Models |
| Section 3.4 — Model Evaluation | Step 4+5: Evaluate + Plots |
| Section 3.5 — XAI (LIME, SHAP) | Step 7: XAI Explanations |

---

## Phân Tích Code

### Imports

```python
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, explore_data
from data_preprocessing import preprocess_data
from models import train_all_models, predict_all_models
from evaluation import evaluate_all_models, print_results_table, generate_all_plots
from explainability import generate_explanations
```

**`sys.path.insert(0, ...)`:**  
Thêm thư mục `main/` vào Python path để có thể import các module cùng cấp (data_loader, models...) bất kể script được gọi từ thư mục nào.

**Tại sao không import tất cả hàm (ví dụ `from models import *`)?**  
Import specific giúp code **rõ ràng**: đọc phần đầu file là biết ngay module nào cung cấp hàm nào.

---

### Setup Thư Mục Kết Quả

```python
results_dir = os.path.join(os.path.dirname(__file__), 'results')
```

Tạo đường dẫn tuyệt đối đến thư mục `results/` kề bên `main.py`:
```
D:\Customer_Churn_Prediction_ML\main\results\
```

Thư mục này được tạo tự động bởi hàm `ensure_results_dir()` trong `evaluation.py` nếu chưa tồn tại.

---

### 7 Bước Pipeline

#### Bước 1: Load Data

```python
df = load_data()
explore_data(df)
```

- `load_data()` → đọc CSV, trả về DataFrame
- `explore_data()` → in thống kê cơ bản ra console

#### Bước 2: Preprocess

```python
data = preprocess_data(df, use_smote=True)
```

- `use_smote=True`: Bật SMOTE để cân bằng dataset
- `data` là dict chứa `X_train`, `X_test`, `y_train`, `y_test`, `feature_names`, ...

**Output thực tế:**
```
✓ Missing values handled          (xử lý 30849 NaN)
✓ Encoded 18 categorical columns
✓ Data split: Train=5282, Test=1761
✓ Features scaled
✓ SMOTE applied: Train size increased to 7760

Preprocessing complete!
  Train set: (7760, 25)
  Test set: (1761, 25)
  Churn rate (train): 50.0%
```

#### Bước 3: Train Models

```python
trained_models = train_all_models(data['X_train'], data['y_train'])
```

- Train cả 5 model (LR, KNN, NB, DT, RF)
- RF sẽ chạy GridSearchCV tự động (~2–5 phút)
- Trả về dict `{tên: model_đã_fit}`

**Output thực tế:**
```
Training Logistic Regression... ✓
Training KNN... ✓
Training Naive Bayes... ✓
Training Decision Tree... ✓
Training Random Forest...
Tuning Random Forest with GridSearchCV...
  Best params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
  Best CV AUC: 0.9621
✓ All 5 models trained successfully!
```

#### Bước 4: Predict

```python
predictions = predict_all_models(trained_models, data['X_test'])
```

- Chạy `model.predict()` và `model.predict_proba()` trên test set
- Trả về dict `{tên: {'y_pred': ..., 'y_prob': ...}}`

#### Bước 5: Evaluate

```python
results = evaluate_all_models(data['y_test'], predictions)
print_results_table(results)
```

- Tính Accuracy, Sensitivity, Specificity, AUC cho từng model
- In bảng so sánh ra console

**Output thực tế:**
```
Logistic Regression:  Acc=77.00%  Sensitivity=83.94%  Specificity=74.50%  AUC=0.8837
KNN:                  Acc=72.63%  Sensitivity=80.30%  Specificity=69.86%  AUC=0.8271
Naive Bayes:          Acc=76.21%  Sensitivity=79.23%  Specificity=75.12%  AUC=0.8478
Decision Tree:        Acc=79.78%  Sensitivity=69.38%  Specificity=83.54%  AUC=0.7918
Random Forest:        Acc=82.28%  Sensitivity=72.16%  Specificity=85.94%  AUC=0.8892

🏆 Best Model: Random Forest (AUC = 0.8892)
```

#### Bước 6: Generate Plots

```python
generate_all_plots(data['y_test'], predictions, results, results_dir)
```

Tạo 3 file PNG:
- `confusion_matrices.png` (Figure 5)
- `roc_curves.png` (Figure 6)
- `metrics_comparison.png` (Figure 7)

#### Bước 7: XAI

```python
rf_model = trained_models['Random Forest']
generate_explanations(
    rf_model,
    data['X_train'],
    data['X_test'],
    data['feature_names'],
    results_dir
)
```

**Tại sao chỉ dùng Random Forest cho XAI?**  
Bài báo (Section 3.5) rõ ràng: *"illustrates the results of the **best** customer churn prediction models"* — Random Forest là mô hình tốt nhất (AUC=0.95).

---

## Luồng Dữ Liệu Tổng Thể

```
CSV file (7043 × 38)
        │
        ▼ load_data()
  DataFrame (7043 × 38)
        │
        ▼ preprocess_data(use_smote=True)
  ┌─────────────────────────────────────┐
  │  X_train (7760 × 25)  y_train (7760,) │  ← sau SMOTE (từ 5282 lên 7760)
  │  X_test  (1761 × 25)  y_test  (1761,) │
  │  feature_names (list, 25 tên)          │
  └──────────┬───────────────────────────-─┘
             │
             ▼ train_all_models()
  trained_models = {
      'Logistic Regression': <fitted LR>,
      'KNN': <fitted KNN>,
      'Naive Bayes': <fitted NB>,
      'Decision Tree': <fitted DT>,
      'Random Forest': <GridSearchCV → max_depth=20, n_estimators=200, CV AUC=0.9621>
  }
             │
             ▼ predict_all_models()
  predictions = {
      'Logistic Regression': {'y_pred': [...], 'y_prob': [...]},
      ...
  }
             │
     ┌───────┴──────────────┐
     │                      │
     ▼ evaluate_all_models() ▼ generate_all_plots()
  results = {              results/
      'RF': {               ├── confusion_matrices.png
          'Accuracy': 0.8228 ├── roc_curves.png
          'AUC': 0.8892,    └── metrics_comparison.png
          ...
      },
      ...
  }
             │
             ▼ generate_explanations(rf_model, ...)
          results/              SHAP values shape: (100, 25, 2)
          ├── lime_sample_0.png
          ├── lime_sample_1.png
          ├── shap_summary.png
          └── shap_bar.png
```

---

## Cách Chạy

```powershell
# 1. Kích hoạt môi trường ảo
cd D:\Customer_Churn_Prediction_ML
.venv\Scripts\Activate.ps1

# 2. Cài thư viện (nếu chưa)
pip install -r requirements.txt

# 3. Chạy pipeline
cd main
python main.py
```

**Thời gian ước tính:**
| Bước | Thời gian |
|---|---|
| Load + Preprocess + SMOTE | ~10s |
| Train LR, KNN, NB, DT | ~30s |
| GridSearchCV RF (120 fits) | ~3–8 phút |
| Evaluation + Plots | ~10s |
| LIME (2 mẫu) | ~30s |
| SHAP (100 mẫu, TreeExplainer) | ~30s |
| **Tổng** | **~5–10 phút** |

---

## Cấu Trúc Hàm `main()`

```python
def main():
    # ...các bước 1-7...
    
    print(f"📁 Results saved to: {results_dir}")
    print("Generated files:")
    print("  - confusion_matrices.png (Figure 5)")
    print("  - roc_curves.png (Figure 6)")
    print("  - metrics_comparison.png (Figure 7)")
    print("  - lime_sample_*.png (Figure 8)")
    print("  - shap_summary.png (Figure 9)")
    print("  - shap_bar.png (Figure 10)")
    
    return results   # ← Trả về dict kết quả để có thể dùng tiếp nếu cần

if __name__ == "__main__":
    results = main()
```

**`if __name__ == "__main__"`:**  
Đảm bảo `main()` chỉ chạy khi file được gọi **trực tiếp** (`python main.py`), không chạy khi được import bởi file khác. Đây là pattern chuẩn của Python.

---

## Tóm Tắt Toàn Bộ Project

```
Bài báo Research Paper                     Code Implementation
─────────────────────────────────────────────────────────────
Section 3.2: Dataset                  →   data_loader.py
             - 7043 rows, 38 columns  →   load_data()        ✓ khớp
             - Missing values: N/A    →   30849 NaN thực tế
             - Train/Test 75:25       →   5282 train / 1761 test ✓

Section 3.3: ML Algorithms            →   models.py
             - Logistic Regression    →   LogisticRegression  ✓
             - KNN                    →   KNeighborsClassifier ✓
             - Naïve Bayes            →   GaussianNB          ✓
             - Decision Tree          →   DecisionTreeClassifier ✓
             - Random Forest          →   RandomForestClassifier + GridSearchCV
                                          Best: max_depth=20, CV AUC=0.9621

Section 3.4: Evaluation               →   evaluation.py
             - Confusion Matrix       →   confusion_matrix() → Figure 5 ✓
             - Accuracy, Sensitivity, →   calculate_metrics()
               Specificity            →   print_results_table()
             - ROC / AUC              →   roc_curve(), auc() → Figure 6 ✓
             - Model comparison       →   metrics_comparison.png (Figure 7) ✓

Section 3.5: XAI                      →   explainability.py
             - LIME                   →   lime.lime_tabular → Figure 8 ✓
             - SHAP                   →   shap.TreeExplainer → Figure 9, 10 ✓
                                          SHAP values shape: (100, 25, 2)

Section 4: Results (bài báo vs code)
             LR:  Acc 75.53% → 77.00%   AUC 0.84 → 0.8837
             KNN: Acc 71.29% → 72.63%   AUC 0.81 → 0.8271
             NB:  Acc 79.84% → 76.21%   AUC 0.88 → 0.8478
             DT:  Acc 80.04% → 79.78%   AUC 0.80 → 0.7918
             RF:  Acc 86.94% → 82.28%   AUC 0.95 → 0.8892  ← vẫn là best
             [Khác biệt do code thêm SMOTE + toàn bộ 7043 mẫu]
```

---

*Trước: [05_explain_explainability.md](05_explain_explainability.md)*  
*Đây là file cuối cùng trong chuỗi giải thích.*
