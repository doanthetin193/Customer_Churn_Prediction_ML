# Giải thích: `evaluation.py` — Bước 4: Đánh Giá Mô Hình

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py (BƯỚC 1)
      └─► data_preprocessing.py (BƯỚC 2)
           └─► models.py (BƯỚC 3)
                └─► [BƯỚC 4] evaluation.py  ← Bạn đang ở đây
                     └─► explainability.py
```

---

## Mục đích

Đo lường hiệu suất của 5 model trên tập test bằng 4 metric chính, vẽ **3 biểu đồ** (Confusion Matrix, ROC Curve, Bar Chart) tương ứng với bài báo Section 3.4 và Figures 5–7.

---

## Liên hệ với Bài Báo (Section 3.4 — Evaluation Metrics)

> *"Four performance metrics were selected: Accuracy, Sensitivity, Specificity, and AUC."*

| Metric trong bài báo | Công thức | Tương ứng trong code |
|---|---|---|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | `accuracy_score(y_true, y_pred)` |
| Sensitivity (Recall/TPR) | $\frac{TP}{TP+FN}$ | `tp / (tp + fn)` |
| Specificity (TNR) | $\frac{TN}{TN+FP}$ | `tn / (tn + fp)` |
| AUC (ROC) | Diện tích dưới ROC curve | `auc(fpr, tpr)` |
| Figure 5 — Confusion Matrix | Bảng TP/TN/FP/FN | `plot_confusion_matrices()` |
| Figure 6 — ROC Curves | Đường ROC từng model | `plot_roc_curves()` |
| Figure 7 — Metrics Comparison | Bar chart so sánh | `plot_metrics_comparison()` |

---

## Phân tích Code Chi Tiết

### Hàm 1: `calculate_metrics(y_true, y_pred, y_prob)`

```python
def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'Accuracy': accuracy, 'Sensitivity': sensitivity,
        'Specificity': specificity, 'AUC': roc_auc,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'fpr': fpr, 'tpr': tpr
    }
```

**Confusion Matrix — 4 giá trị cơ bản:**

```
                    Dự đoán:     Stayed   Churned
                                (0)      (1)
Thực tế: Stayed (0)    │  TN  │  FP  │
         Churned (1)   │  FN  │  TP  │
```

| Ký hiệu | Ý nghĩa trong bài toán churn |
|---|---|
| **TP** (True Positive) | Dự đoán Churn đúng → phát hiện đúng khách sẽ rời đi |
| **TN** (True Negative) | Dự đoán Stayed đúng → phát hiện đúng khách trung thành |
| **FP** (False Positive) | Dự đoán Churn sai → tốn chi phí retention vô ích |
| **FN** (False Negative) | Dự đoán Stayed sai → **nguy hiểm nhất**: bỏ sót khách Churn |

**`confusion_matrix(y_true, y_pred).ravel()`:**

`.ravel()` biến ma trận 2×2 thành array 1D theo thứ tự `[TN, FP, FN, TP]` — tiết kiệm 4 dòng code index thủ công.

**Công thức 4 metric:**

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

$$Sensitivity = \frac{TP}{TP + FN}$$ — Trong 1869 khách Churn, model phát hiện được bao nhiêu?

$$Specificity = \frac{TN}{TN + FP}$$ — Trong 4720 khách Stayed, model giữ đúng bao nhiêu?

$$AUC = \int_0^1 TPR(FPR) \, d(FPR)$$ — Diện tích dưới ROC, đo khả năng phân biệt tổng thể

**`roc_curve(y_true, y_prob)`:** Trả về 3 arrays:
- `fpr`: False Positive Rate tại từng ngưỡng threshold (0→1)
- `tpr`: True Positive Rate (Sensitivity) tại từng ngưỡng
- `_` (thresholds): không cần dùng → bỏ qua với `_`

---

### Hàm 2: `evaluate_all_models(y_test, predictions)`

```python
def evaluate_all_models(y_test, predictions) -> dict:
    results = {}
    for name, pred in predictions.items():
        metrics = calculate_metrics(y_test, pred['y_pred'], pred['y_prob'])
        results[name] = metrics
        print(f"\n{name}:")
        print(f"  Accuracy:    {metrics['Accuracy']*100:.2f}%")
        ...
    return results
```

**Input `predictions` dict:**

```python
predictions = {
    'Logistic Regression': {'y_pred': array([0,1,0,...]), 'y_prob': array([0.23, 0.87, ...])},
    'KNN':                 {'y_pred': ..., 'y_prob': ...},
    ...
}
```

Loop qua từng model → gọi `calculate_metrics()` → lưu vào `results` dict.

**Kết quả thực tế:**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 80.46% | 86.72% | 77.98% | 0.9112 |
| KNN | 81.01% | 66.81% | 86.62% | 0.8521 |
| Naive Bayes | 80.10% | 78.59% | 80.69% | 0.8771 |
| Decision Tree | 82.10% | 68.09% | 87.64% | 0.7787 |
| **Random Forest** | **86.65%** | **74.52%** | **91.45%** | **0.9264** |

---

### Hàm 3: `print_results_table(results)`

```python
def print_results_table(results):
    ...
    best_model = max(results.items(), key=lambda x: x[1]['AUC'])
    print(f"🏆 Best Model: {best_model[0]} (AUC = {best_model[1]['AUC']:.4f})")
```

`max(results.items(), key=lambda x: x[1]['AUC'])` — duyệt qua dict items, key là hàm lấy giá trị AUC → tìm model có AUC cao nhất.

**Bài báo cũng kết luận RF là best:**
> *"...the RF algorithm provides better performance in terms of accuracy and AUC."*

---

### Hàm 4: `plot_confusion_matrices()` — Figure 5

```python
def plot_confusion_matrices(y_test, predictions, results_dir='results'):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, pred['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        axes[idx].set_title(f'{name}')
    
    axes[-1].axis('off')  # Ẩn subplot thứ 6 (không có model nào)
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), ...)
```

**Grid 2×3 = 6 subplot cho 5 model:**

```
[ LR       ]  [ KNN      ]  [ NB       ]
[ DT       ]  [ RF       ]  [ (trống)  ]
```

**`axes.flatten()`:** `subplots(2, 3)` trả về array 2D shape (2,3) → `flatten()` biến thành 1D (6,) để loop bằng index đơn giản.

**`axes[-1].axis('off')`:** Ẩn subplot cuối (index 5) vì chỉ có 5 model, không cần hiển thị trống.

**`sns.heatmap` parameters:**
- `annot=True`: Hiện số trong ô
- `fmt='d'`: Format integer (không phải float)
- `cmap='Blues'`: Màu xanh dương, đậm = số lớn

---

### Hàm 5: `plot_roc_curves()` — Figure 6

```python
def plot_roc_curves(results, results_dir='results'):
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    for idx, (name, metrics) in enumerate(results.items()):
        plt.plot(metrics['fpr'], metrics['tpr'],
                 color=colors[idx], lw=2,
                 label=f"{name} (AUC = {metrics['AUC']:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')
```

**Đường chéo `[0,1],[0,1]`:** Biểu diễn classifier ngẫu nhiên (50% đoán đúng), AUC = 0.5. Model tốt phải nằm **trên** đường chéo này.

**ROC Curve ý nghĩa:**

- Trục X: FPR (= 1 - Specificity) → càng thấp càng tốt
- Trục Y: TPR (= Sensitivity) → càng cao càng tốt
- AUC = 1.0: perfect classifier; AUC = 0.5: random; AUC < 0.5: tệ hơn random

**Random Forest có ROC curve gần góc trên-trái nhất** → AUC = 0.9264.

---

### Hàm 6: `plot_metrics_comparison()` — Figure 7

```python
def plot_metrics_comparison(results, results_dir='results'):
    metrics_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    x = np.arange(len(models))
    width = 0.2

    for i, metric in enumerate(metrics_names):
        values = [results[m][metric] for m in models]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])
```

**Grouped bar chart:**
- 5 nhóm thanh (1 nhóm/model) trên trục X
- 4 metric/nhóm (Accuracy, AUC, Sensitivity, Specificity) — 4 màu khác nhau
- `offset = (i - 1.5) * width`: dịch chuyển từng thanh để không chồng lên nhau

**`width=0.2` × 4 metrics = 0.8** → thanh của 4 metrics chiếm 0.8 đơn vị, để khoảng trống 0.2 giữa các nhóm model.

---

### Hàm 7: `generate_all_plots()` — Wrapper tổng

```python
def generate_all_plots(y_test, predictions, results, results_dir='results'):
    plot_confusion_matrices(y_test, predictions, results_dir)
    plot_roc_curves(results, results_dir)
    plot_metrics_comparison(results, results_dir)
```

Gọi lần lượt 3 hàm trên. Từ `main.py`:

```python
generate_all_plots(data['y_test'], predictions, results, results_dir)
```

---

## Files output được tạo ra

| File | Kích thước | Tương ứng với |
|---|---|---|
| `results/confusion_matrices.png` | 15×10 inch, 150 DPI | Figure 5 trong bài báo |
| `results/roc_curves.png` | 10×8 inch, 150 DPI | Figure 6 trong bài báo |
| `results/metrics_comparison.png` | 12×6 inch, 150 DPI | Figure 7 trong bài báo |

**`matplotlib.use('Agg')` ở đầu file:**

Backend `Agg` là non-interactive — render file PNG mà không mở cửa sổ GUI. Cần thiết vì code chạy từ terminal (không có display server). Nếu dùng `plt.show()` với Agg → không hiển thị, nhưng `plt.savefig()` vẫn hoạt động bình thường.

---

## Tại sao chọn 4 metric này?

Bài báo giải thích:

> *"Accuracy measures the overall correctness of the model. Sensitivity measures the proportion of actual positives correctly identified — important for not missing churning customers. Specificity measures the proportion of actual negatives correctly identified — important to avoid disturbing loyal customers unnecessarily. AUC provides a threshold-independent measure of discrimination ability."*

Trong bài toán churn:
- **FN nguy hiểm nhất** (miss churner) → muốn Sensitivity cao
- **FP tốn kém** (can thiệp vào khách trung thành) → muốn Specificity cao
- Cả 4 metric cùng nhau cho cái nhìn toàn diện, không bị bản tóm tắt bởi 1 con số đơn lẻ
