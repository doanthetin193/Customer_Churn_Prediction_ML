# Giải thích: `main.py` — Điểm Khởi Động Pipeline DL

## Vị trí trong Pipeline

```
[ENTRY POINT] main.py (DL)  ← Bạn đang ở đây
 ├─► BƯỚC 1: data_loader.py
 ├─► BƯỚC 2: data_preprocessing.py
 ├─► BƯỚC 3: models_dl.py   (build_mlp + train_mlp + find_optimal_threshold)
 └─► BƯỚC 4: evaluation.py  (calculate_metrics + generate_all_outputs)
```

---

## Mục đích

Điều phối toàn bộ DL pipeline end-to-end — từ load dữ liệu đến lưu model. Khác với ML pipeline (7 bước), DL pipeline có **thêm bước tách validation set** và **tìm threshold tối ưu**.

---

## So sánh cấu trúc với ML Pipeline

| Bước | ML `main/main.py` | DL `main_DL/main.py` |
|---|---|---|
| 1. Load data | `load_data()` | `load_data()` |
| 2. Preprocess | `preprocess_data(use_smote=False)` | `preprocess_data(use_smote=False)` |
| **3. Val split** | ❌ Không có | ✅ `train_test_split(test=0.2)` từ train |
| **4. Class weights** | ❌ Trong model def | ✅ `compute_class_weight()` → truyền vào `model.fit()` |
| 5. Train | `train_all_models(tune_rf=False)` | `train_mlp(epochs=150, patience=20)` |
| **6. Optimal threshold** | ❌ Dùng 0.5 | ✅ `find_optimal_threshold()` trên val set |
| 7. Predict | `predict_all_models()` | `predict_mlp(threshold=0.54)` |
| 8. Evaluate | `evaluate_all_models()` | `calculate_metrics()` |
| 9. Plots | 3 plots (confusion, ROC, bar) | 3 plots (confusion, ROC, training history) |
| **10. Save model** | ❌ Không lưu | ✅ `model.save()` → `.keras` file |
| **11. Save JSON** | ❌ Không | ✅ `metrics_dl.json` |

---

## Phân tích Code Chi Tiết

### Header và Import

```python
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import explore_data, load_data
from data_preprocessing import preprocess_data
from evaluation import calculate_metrics, generate_all_outputs, print_metrics
from models_dl import find_optimal_threshold, predict_mlp, train_mlp
```

**Imports đáng chú ý:**

| Import | Lý do |
|---|---|
| `train_test_split` | Cần thêm lần nữa để tạo val set (data_preprocessing đã split train/test rồi) |
| `compute_class_weight` | Tính class weights tự động từ sklearn |
| `find_optimal_threshold` | Hàm tìm threshold Youden's J — không có tương đương trong ML pipeline |

---

### Hàm `main()` — Pipeline 6 Bước

#### Bước 1: Tải dữ liệu

```python
print("\nSTEP 1: Loading data")
df = load_data()
explore_data(df)
```

Giống ML pipeline. `df` = DataFrame thô 7043×38.

---

#### Bước 2: Tiền xử lý

```python
use_smote = False  # Class weight thay thế SMOTE
print("\nSTEP 2: Preprocessing data")
data = preprocess_data(df, use_smote=use_smote)
```

Kết quả:
- `data["X_train"]`: (4941, 25) — sẽ bị split thêm
- `data["X_test"]`:  (1648, 25) — giữ nguyên không đụng đến

---

#### Bước 3: Tách validation set (DL-specific)

```python
print("\nSTEP 3: Building train/validation split")
X_train, X_val, y_train, y_val = train_test_split(
    data["X_train"],
    data["y_train"],
    test_size=0.2,
    random_state=42,
    stratify=data["y_train"],  # Giữ tỉ lệ churn 28.4%
)
```

**Tại sao Neural Network cần validation set riêng?**

| Mục đích | Giải thích |
|---|---|
| **EarlyStopping** | Monitor `val_auc` để biết khi nào dừng training → tránh overfit |
| **ReduceLROnPlateau** | Giảm lr khi `val_auc` plateau |
| **`find_optimal_threshold()`** | Tìm threshold tối ưu trên val (không dùng test để tránh leakage) |
| **`restore_best_weights`** | Keras lưu weights epoch có val_auc tốt nhất → cần val để biết "tốt nhất" |

**Sklearn ML không cần** vì không có quá trình training iterative — `.fit()` là 1 bước.

**Phân phối data sau 2 lần split:**

```
Toàn bộ (6589 mẫu)
    ├── Train 75% (4941 mẫu)
    │    ├── Train final 80% (3953 mẫu)  ← X_train sau split
    │    └── Validation   20% (988 mẫu)  ← X_val
    └── Test 25% (1648 mẫu)              ← data["X_test"] — không đụng
```

---

#### Bước 4: Tính class weights

```python
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_train,                # Tính từ FINAL train set (3953 mẫu)
)
class_weight_dict = {
    0: float(class_weights[0]),
    1: float(class_weights[1]),
}
print(f"Class weights: {class_weight_dict}")
```

**Tại sao tính từ `y_train` (sau val split) chứ không phải `data["y_train"]`?**

`compute_class_weight` phải tính dựa trên tập data thực sự đưa vào model training (3953 mẫu). Dùng `data["y_train"]` (4941 mẫu, bao gồm cả val) → weights sai lệch một chút vì bao gồm samples model không thực sự train trên đó.

**Công thức compute_class_weight:**

$$w_k = \frac{n\_samples}{n\_classes \times n\_samples\_k}$$

$$w_0 = \frac{3953}{2 \times 2826} \approx 0.699 \text{ (Stayed)}$$
$$w_1 = \frac{3953}{2 \times 1127} \approx 1.754 \text{ (Churned)}$$

**Kết quả in ra:**
```
Class weights: {0: 0.698, 1: 1.763}
```

Mỗi sample Churned có weight gấp ~2.5× sample Stayed trong loss function.

---

#### Bước 5: Training MLP

```python
print("\nSTEP 4: Training MLP")
model, history = train_mlp(
    X_train, y_train,      # Train final: 3953 mẫu
    X_val, y_val,          # Validation:  988 mẫu
    class_weight=class_weight_dict,
    epochs=150,
    batch_size=64,
    patience=20,
)
```

**Tham số training:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `epochs=150` | 150 | Tối đa 150 lần duyệt qua train data (EarlyStopping sẽ dừng sớm hơn) |
| `batch_size=64` | 64 | Số mẫu mỗi gradient update. 3953/64 ≈ 62 batches/epoch |
| `patience=20` | 20 | Cho phép 20 epoch không cải thiện trước khi dừng |

**`batch_size=64` — tại sao không dùng toàn bộ data?**

Mini-batch gradient descent (batch 64):
- Cập nhật weights 62 lần/epoch → học nhanh hơn
- Noise trong gradient từ small batch → tránh local minima
- Phù hợp với GPU memory (CPU trong project này)

Full-batch (batch = 3953):
- 1 lần update/epoch → chậm
- Gradient chính xác hơn nhưng dễ kẹt local minima

---

#### Bước 5b: Tìm threshold tối ưu

```python
print("\nSTEP 4b: Finding optimal classification threshold on validation set")
optimal_threshold, youden_j = find_optimal_threshold(model, X_val, y_val)
```

Scan 90 threshold (0.05 → 0.94), tìm $t^*$ maximize Youden's J:

$$t^* = \arg\max_t \left[ \text{Sensitivity}(t) + \text{Specificity}(t) - 1 \right]$$

**Kết quả: `optimal_threshold = 0.54, youden_j = 0.7472`**

Sử dụng **validation set** (988 mẫu) để tìm threshold, không phải test set → tránh data leakage vào final evaluation.

---

#### Bước 6: Evaluate trên test set

```python
print("\nSTEP 5: Evaluating on test set")
y_pred, y_prob = predict_mlp(model, data["X_test"], threshold=optimal_threshold)
metrics = calculate_metrics(data["y_test"], y_pred, y_prob)
metrics["threshold"] = optimal_threshold
metrics["youden_j"] = youden_j
print_metrics(metrics)
print(f"  Threshold used: {optimal_threshold:.2f}")
```

**Thêm 2 key vào dict metrics:**

```python
metrics["threshold"] = 0.54   # Lưu lại threshold đã dùng
metrics["youden_j"]  = 0.7472 # Lưu lại J score của threshold đó
```

Hai giá trị này sẽ được lưu vào `metrics_dl.json` qua `generate_all_outputs()`.

**Prediction với threshold 0.54:**

```python
y_pred = (y_prob >= 0.54).astype(int)
# P(Churn) >= 0.54 → predict Churn
# P(Churn)  < 0.54 → predict Stayed
```

**Kết quả:**

```
==================================================
DL MODEL EVALUATION
==================================================
Accuracy:    81.74%
Sensitivity: 84.37%
Specificity: 80.69%
AUC:         0.9132
  Threshold used: 0.54
```

---

#### Bước 7: Lưu artifacts

```python
print("\nSTEP 6: Saving artifacts")
generate_all_outputs(data["y_test"], y_pred, metrics, history, results_dir)

# Lưu model
model_path = os.path.join(results_dir, "mlp_churn_model.keras")
model.save(model_path)
print(f"Saved model to: {model_path}")
```

**`model.save()` — định dạng `.keras`:**

Format `.keras` (Keras v3 native format) lưu toàn bộ model:
- Kiến trúc (config)
- Weights (trained parameters ~49k)
- Optimizer state
- Compile config (loss, metrics)

Có thể load lại bằng:

```python
loaded_model = keras.models.load_model("mlp_churn_model.keras")
y_pred, y_prob = predict_mlp(loaded_model, X_new, threshold=0.54)
```

**Tại sao lưu model?** Để deploy hoặc tái sử dụng mà không cần train lại (~1-2 phút training).

---

## Sơ đồ luồng dữ liệu tổng thể

```
telecom_customer_churn.csv
        │
        ▼
 load_data()                          → DataFrame(7043×38)
        │
        ▼
 preprocess_data(use_smote=False)
        │ X_train(4941×25), X_test(1648×25)
        │ y_train(4941,),   y_test(1648,)
        ▼
 train_test_split(test=0.2)           → tách val từ train
        │ X_train(3953×25), X_val(988×25), y_train(3953,), y_val(988,)
        ▼
 compute_class_weight()               → {0: 0.698, 1: 1.763}
        │
        ▼
 train_mlp(epochs=150, patience=20)   → model (trained) + history
        │
        ▼
 find_optimal_threshold(X_val)        → threshold=0.54, J=0.7472
        │
        ▼
 predict_mlp(X_test, threshold=0.54)  → y_pred(1648,), y_prob(1648,)
        │
        ├──────────────┬──────────────────────┐
        ▼              ▼                      ▼
 confusion_matrix   roc_curve_dl.png   training_history_dl.png
 _dl.png            metrics_dl.json    mlp_churn_model.keras
```

---

## Thời gian chạy ước tính

| Bước | Thời gian |
|---|---|
| Load + Preprocess | ~2 giây |
| Val split + Class weights | < 1 giây |
| Training MLP (CPU, ~70-90 epochs thực tế) | ~3-5 phút |
| Find threshold | ~5 giây (predict 988 mẫu × 90 thresholds) |
| Evaluate + Save plots | ~5 giây |
| Save model | ~1 giây |
| **Tổng** | **~5-7 phút** |

---

## Các quyết định thiết kế quan trọng

| Quyết định | Lý do |
|---|---|
| `use_smote=False` | Khớp phương pháp bài báo, tránh tạo dữ liệu giả |
| Val split 20% từ train | Neural network cần validation để monitor training |
| `class_weight` thay SMOTE | Điều chỉnh loss weight trực tiếp, sạch hơn về mặt thống kê |
| `optimal_threshold=0.54` | Youden's J trên val → cân bằng Sensitivity/Specificity |
| `epochs=150, patience=20` | Cho model đủ thời gian học nhưng EarlyStopping dừng kịp thời |
| Lưu `.keras` | Tái sử dụng model không cần train lại |
| Lưu `metrics_dl.json` | So sánh kết quả với các run khác, dễ đọc bằng script |
