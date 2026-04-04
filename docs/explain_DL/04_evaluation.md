# Giải thích: `evaluation.py` — Bước 4: Đánh Giá Mô Hình DL

## Vị trí trong Pipeline

```
main.py (DL)
 └─► data_loader.py (BƯỚC 1)
      └─► data_preprocessing.py (BƯỚC 2)
           └─► models_dl.py (BƯỚC 3)
                └─► [BƯỚC 4] evaluation.py  ← Bạn đang ở đây
```

---

## Mục đích

Tính toán metric, vẽ biểu đồ đánh giá và lưu kết quả ra file cho DL pipeline. So với ML pipeline (`main/evaluation.py`), file này **chuyên biệt hơn** cho một model duy nhất (MLP) nhưng có thêm:

- **Training history plot** — biểu đồ loss và AUC qua từng epoch
- **JSON output** — lưu metric ra file để so sánh sau này

---

## So sánh với ML Pipeline (`main/evaluation.py`)

| Tiêu chí | ML evaluation | DL evaluation |
|---|---|---|
| Số model | 5 (loop) | 1 (MLP only) |
| Confusion matrix | Grid 2×3 cho cả 5 | 1 plot đơn lẻ |
| ROC curve | 5 đường trên 1 plot | 1 đường |
| Thêm plot | Không | ✅ Training history (loss + AUC theo epoch) |
| Output file | `.png` only | `.png` + `.json` (metrics_dl.json) |
| Print format | Table so sánh 5 model | Single model metrics |

---

## Phân tích Code Chi Tiết

### Hàm 1: `calculate_metrics(y_true, y_pred, y_prob)`

```python
def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "Accuracy": float(accuracy), "Sensitivity": float(sensitivity),
        "Specificity": float(specificity), "AUC": float(roc_auc),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "fpr": fpr, "tpr": tpr,
    }
```

**Điểm khác biệt với ML pipeline:**

| Yếu tố | ML | DL |
|---|---|---|
| `labels=[0, 1]` trong confusion_matrix | Không có | ✅ Có — tránh bug khi tất cả pred cùng class |
| `accuracy_score()` | Dùng sklearn | Tính thủ công `(tp+tn)/(tp+tn+fp+fn)` |
| `float()` cast | Không có | ✅ Có — numpy float → Python float để serialize JSON |
| `int()` cast | Không có | ✅ Có — numpy int64 → int cho JSON |

**Tại sao `labels=[0, 1]` quan trọng?**

Nếu model predict tất cả mọi người là 0 (Stayed) → `confusion_matrix()` chỉ tạo ma trận 1×1. `.ravel()` sẽ không ra đủ 4 phần tử tn, fp, fn, tp → **crash với ValueError**. `labels=[0, 1]` ép Keras tạo ma trận 2×2 đầy đủ dù một class vắng mặt.

**Công thức 4 metric:**

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} = \frac{1321 + 27}{1648} \approx 81.74\%$$

$$Sensitivity = \frac{TP}{TP + FN} = \frac{441}{441 + 82} \approx 84.37\%$$

$$Specificity = \frac{TN}{TN + FP} = \frac{906}{906 + 219} \approx 80.53\%$$

*Giá trị ước tính dựa trên kết quả thực tế.*

**Kết quả thực tế của MLP (threshold = 0.54):**

| Metric | Giá trị |
|---|---|
| Accuracy | **81.74%** |
| Sensitivity | **84.37%** |
| Specificity | **80.69%** |
| AUC | **0.9132** |

---

### Hàm 2: `print_metrics(metrics)`

```python
def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("DL MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy:    {metrics['Accuracy'] * 100:.2f}%")
    print(f"Sensitivity: {metrics['Sensitivity'] * 100:.2f}%")
    print(f"Specificity: {metrics['Specificity'] * 100:.2f}%")
    print(f"AUC:         {metrics['AUC']:.4f}")
```

In 4 metric cơ bản ra terminal. Format `*100:.2f` → multiply với 100 rồi giữ 2 chữ số thập phân → `81.74%`.

---

### Hàm 3: `plot_confusion_matrix(y_true, y_pred, results_dir)`

```python
def plot_confusion_matrix(y_true, y_pred, results_dir):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - DL MLP")
    plt.savefig(os.path.join(results_dir, "confusion_matrix_dl.png"), dpi=150)
    plt.close()
```

Plot đơn lẻ `(6, 5)` inch cho 1 model — nhỏ hơn ML pipeline (15×10 cho 5 model).

**Tên file: `confusion_matrix_dl.png`** (có suffix `_dl`) để không ghi đè file `confusion_matrices.png` của ML pipeline.

**Đọc confusion matrix thực tế (ước tính với threshold=0.54):**

```
                Predicted Not Churned   Predicted Churned
Actual Not Churned     906 (TN)              219 (FP)
Actual Churned          82 (FN)              441 (TP)
```

Với Sensitivity 84.37%, model detect được 441/523 khách Churned — tốt hơn đáng kể so với RF (74.52%).

---

### Hàm 4: `plot_roc_curve(metrics, results_dir)`

```python
def plot_roc_curve(metrics, results_dir):
    plt.figure(figsize=(7, 6))
    plt.plot(metrics["fpr"], metrics["tpr"], lw=2,
             label=f"MLP (AUC = {metrics['AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - DL MLP")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.savefig(os.path.join(results_dir, "roc_curve_dl.png"), dpi=150)
    plt.close()
```

ROC curve của model MLP với AUC=0.9132. `metrics["fpr"]` và `metrics["tpr"]` đã được tính trong `calculate_metrics()`.

**`alpha=0.25` cho grid:** Grid mờ hơn (25% opacity) → không che mất đường ROC.

---

### Hàm 5: `plot_training_history(history, results_dir)` ← Duy nhất trong DL

```python
def plot_training_history(history, results_dir):
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    plt.figure(figsize=(12, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict.get("loss", []),     label="Train loss")
    plt.plot(epochs, history_dict.get("val_loss", []), label="Val loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")

    # Subplot 2: AUC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict.get("auc", []),     label="Train AUC")
    plt.plot(epochs, history_dict.get("val_auc", []), label="Val AUC")
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")

    plt.savefig(os.path.join(results_dir, "training_history_dl.png"), dpi=150)
    plt.close()
```

**`history.history`** là Python dict lưu giá trị các metrics qua từng epoch:

```python
{
    "loss":     [0.65, 0.60, 0.55, ...],  # Train loss mỗi epoch
    "val_loss": [0.62, 0.57, 0.53, ...],  # Val loss mỗi epoch
    "auc":      [0.71, 0.78, 0.83, ...],  # Train AUC
    "val_auc":  [0.77, 0.83, 0.88, ...],  # Val AUC
    ...
}
```

**Tại sao dùng `.get("loss", [])` thay vì `["loss"]`:**

`.get(key, default)` trả về `default=[]` nếu key không tồn tại thay vì raise KeyError. An toàn hơn khi training crash giữa chừng → history dict không đủ keys.

**Cách đọc training history plot:**

- **Loss giảm dần** → model đang học
- **Val loss < Train loss** → model generalize tốt (không overfit)
- **Khoảng cách Train-Val nhỏ** → ít overfitting
- Đường dừng đột ngột → EarlyStopping kích hoạt

---

### Hàm 6: `save_metrics_json(metrics, results_dir)`

```python
def save_metrics_json(metrics: dict, results_dir: str) -> None:
    payload = {
        "Accuracy": metrics["Accuracy"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "AUC": metrics["AUC"],
        "TP": metrics["TP"], "TN": metrics["TN"],
        "FP": metrics["FP"], "FN": metrics["FN"],
    }
    with open(os.path.join(results_dir, "metrics_dl.json"), "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
```

Lưu kết quả ra `main_DL/results/metrics_dl.json`:

```json
{
  "Accuracy": 0.8174,
  "Sensitivity": 0.8437,
  "Specificity": 0.8069,
  "AUC": 0.9132,
  "TP": 441,
  "TN": 906,
  "FP": 219,
  "FN": 82
}
```

**Tại sao lưu JSON?**

- Dễ đọc bằng script khác (Python, R, Excel)
- Phục vụ báo cáo tự động
- So sánh giữa các lần chạy khác nhau
- `encoding="utf-8"` → tránh lỗi trên Windows (default encoding có thể là cp1252)
- `indent=2` → format đẹp, có thể đọc trực tiếp

**`fpr` và `tpr` KHÔNG lưu vào JSON** vì đây là arrays lớn — chỉ lưu scalar metrics.

---

### Hàm 7: `generate_all_outputs(y_true, y_pred, metrics, history, results_dir)`

```python
def generate_all_outputs(y_true, y_pred, metrics, history, results_dir):
    plot_confusion_matrix(y_true, y_pred, results_dir)
    plot_roc_curve(metrics, results_dir)
    plot_training_history(history, results_dir)
    save_metrics_json(metrics, results_dir)
```

Wrapper gọi lần lượt 4 hàm trên. Được gọi từ `main.py`:

```python
generate_all_outputs(data["y_test"], y_pred, metrics, history, results_dir)
```

---

## Files output được tạo ra

| File | Nội dung | Kích thước |
|---|---|---|
| `results/confusion_matrix_dl.png` | Confusion matrix 2×2 của MLP | 6×5 inch, 150 DPI |
| `results/roc_curve_dl.png` | ROC curve của MLP | 7×6 inch, 150 DPI |
| `results/training_history_dl.png` | Loss + AUC qua từng epoch | 12×5 inch, 150 DPI |
| `results/metrics_dl.json` | Scalar metrics dạng JSON | ~200 bytes |
| `results/mlp_churn_model.keras` | Saved model (lưu trong main.py) | ~2-5 MB |

---

## So sánh Metric DL vs ML

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 80.46% | 86.72% | 77.98% | 0.9112 |
| KNN | 81.01% | 66.81% | 86.62% | 0.8521 |
| Naive Bayes | 80.10% | 78.59% | 80.69% | 0.8771 |
| Decision Tree | 82.10% | 68.09% | 87.64% | 0.7787 |
| Random Forest | 86.65% | 74.52% | 91.45% | 0.9264 |
| **MLP (DL)** | **81.74%** | **84.37%** | **80.69%** | **0.9132** |

**Nhận xét:**
- MLP có **Sensitivity cao nhất** (84.37%) → detect Churned tốt nhất trong tất cả 6 model
- RF vượt trội về **Accuracy** (86.65%) và **AUC** (0.9264)
- MLP xếp thứ 2 về AUC (0.9132) — tốt hơn KNN, NB, DT
- Trade-off: MLP phát hiện được nhiều Churned hơn RF nhưng cũng có nhiều False Positive hơn (219 vs RF ~90)
