# Giải thích: `evaluation.py` — Bước 4 & 5: Đánh Giá & Vẽ Biểu Đồ

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py  [✓]
      └─► data_preprocessing.py  [✓]
           └─► models.py  [✓]
                └─► [BƯỚC 4+5] evaluation.py  ← Bạn đang ở đây
                     └─► explainability.py
```

---

## Tổng Quan

File này thực hiện toàn bộ **đánh giá hiệu năng** của 5 mô hình và tạo ra **3 loại biểu đồ** tương ứng với Figure 5, 6, 7 trong bài báo.

**Đầu vào:** `y_test` (nhãn thực), `predictions` (dict dự đoán từng model)  
**Đầu ra:** Dictionary kết quả metrics + 3 file PNG trong `results/`

---

## Liên Hệ Bài Báo (Section 3.4 & 4.2 & 4.3)

| File tạo ra | Tương ứng trong bài báo |
|---|---|
| `confusion_matrices.png` | **Figure 5** — Confusion Matrix values for the five algorithms |
| `roc_curves.png` | **Figure 6** — ROC curves for the five models |
| `metrics_comparison.png` | **Figure 7** — Model comparison (Accuracy, AUC, Sensitivity, Specificity) |
| Console output (bảng số) | **Table 3** — Table of descriptive statistics |

---

## Phần 1: Các Chỉ Số Đánh Giá

### `calculate_metrics(y_true, y_pred, y_prob)`

```python
def calculate_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy    = accuracy_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)    # True Positive Rate
    specificity = tn / (tn + fp)    # True Negative Rate
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    return {'Accuracy': accuracy, 'Sensitivity': sensitivity, ...}
```

### Confusion Matrix — Nền tảng của mọi chỉ số

Bài báo (Section 3.4.1, Table 1):

```
                    Predicted: NOT Churn    Predicted: Churn
Actual: NOT Churn        TN                      FP
Actual: Churn            FN                      TP
```

**Ý nghĩa từng ô (bài báo giải thích):**

| Ký hiệu | Ý nghĩa trong bài toán churn | Hậu quả kinh doanh |
|---|---|---|
| **TP** (True Positive) | Đúng là sẽ churn, dự đoán churn | Tốt: can thiệp đúng người |
| **TN** (True Negative) | Không churn, dự đoán không churn | Tốt: không can thiệp không cần thiết |
| **FP** (False Positive) | Không churn, nhưng dự đoán churn | Xấu: tốn chi phí giữ chân người không có ý định rời bỏ |
| **FN** (False Negative) | Thực ra churn, nhưng dự đoán không churn | **Nguy hiểm nhất**: bỏ lỡ khách hàng, mất doanh thu |

---

### Công Thức Các Chỉ Số

**Bài báo (Section 3.4.2)** định nghĩa 4 chỉ số chính:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

> Tỷ lệ dự đoán đúng tổng thể

$$\text{Sensitivity} = \frac{TP}{TP + FN}$$

> = Recall = True Positive Rate — Khả năng phát hiện đúng khách hàng sẽ churn  
> **Quan trọng:** FN thấp → Sensitivity cao → Ít bỏ sót churner

$$\text{Specificity} = \frac{TN}{TN + FP}$$

> = True Negative Rate — Khả năng xác định đúng người không churn  
> FP thấp → Specificity cao → Ít báo nhầm

$$\text{AUC} = \text{Area Under ROC Curve}$$

> Đo lường tổng thể, không phụ thuộc ngưỡng phân loại. AUC = 1.0 là hoàn hảo.

---

### Bảng Kết Quả Thực Tế (khi chạy code với SMOTE)

```
======================================================================
MODEL COMPARISON TABLE
======================================================================
Model                  Accuracy  Sensitivity  Specificity       AUC
----------------------------------------------------------------------
Logistic Regression      77.00%       83.94%       74.50%    0.8837
KNN                      72.63%       80.30%       69.86%    0.8271
Naive Bayes              76.21%       79.23%       75.12%    0.8478
Decision Tree            79.78%       69.38%       83.54%    0.7918
Random Forest            82.28%       72.16%       85.94%    0.8892
----------------------------------------------------------------------

🏆 Best Model: Random Forest (AUC = 0.8892)
```

### So Sánh với Bài Báo (không dùng SMOTE)

| Model | Acc (bài báo) | Acc (code) | AUC (bài báo) | AUC (code) | Âm lượng khác biệt |
|---|---|---|---|---|---|
| Logistic Regression | 75.53% | **77.00%** | 0.84 | **0.8837** | Code tốt hơn nhờ SMOTE tăng tính đa dạng train |
| KNN | 71.29% | **72.63%** | 0.81 | **0.8271** | Tương đương |
| Naive Bayes | 79.84% | 76.21% | 0.88 | 0.8478 | Bài báo tốt hơn — NB nhạy với phân phối |
| Decision Tree | 80.04% | 79.78% | 0.80 | 0.7918 | Tương đương |
| **Random Forest** | 86.94% | 82.28% | 0.95 | 0.8892 | Bài báo cao hơn do không SMOTE, test set cân bằng hơn |

> **Lý do sự khác biệt:** SMOTE cân bằng train set (→ Sensitivity tăng), nhưng test set vẫn giữ phân phối thực (67% Not Churn, 33% Churn). Mô hình bây giờ có xu hướng dự đoán Churn nhiều hơn → Sensitivity cao hơn, Accuracy thấp hơn so với bài báo.

---

## Phần 2: Vẽ Biểu Đồ

### Hàm `plot_confusion_matrices()` → Figure 5

```python
def plot_confusion_matrices(y_test, predictions, results_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Grid 2×3 (5 model + 1 trống)
    
    for idx, (name, pred) in enumerate(predictions.items()):
        cm = confusion_matrix(y_test, pred['y_pred'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
```

**Giải thích tham số:**

| Tham số | Ý nghĩa |
|---|---|
| `annot=True` | Hiển thị số trong từng ô |
| `fmt='d'` | Format số nguyên (không dấu phẩy) |
| `cmap='Blues'` | Màu xanh: số lớn = màu đậm |
| `plt.subplots(2, 3)` | 6 subplot, 5 model + 1 ẩn |

---

### Hàm `plot_roc_curves()` → Figure 6

```python
def plot_roc_curves(results, results_dir):
    for idx, (name, metrics) in enumerate(results.items()):
        plt.plot(
            metrics['fpr'],    # False Positive Rate (trục X)
            metrics['tpr'],    # True Positive Rate  (trục Y)
            label=f"{name} (AUC = {metrics['AUC']:.4f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
```

**Giải thích ROC Curve:**

```
TPR ^ 
1.0 ─┤              ╭────── Đường hoàn hảo (AUC=1.0)
     │           ╭──╯
     │        ╭──╯   ← RF (AUC=0.95)
     │      ╭─╯   ← NB (AUC=0.88)
     │    ╭─╯
0.5 ─├───╯─────────── Đường ngẫu nhiên (AUC=0.50)
     │  ╱
0.0 ─┤──────────────► FPR
     0.0    0.5    1.0
```

- **Đường cong càng gần góc trái trên** → mô hình càng tốt
- **AUC** = diện tích dưới đường cong → đo tổng thể khả năng phân biệt 2 class
- Đường chéo `--` = dự đoán ngẫu nhiên (baseline)

---

### Hàm `plot_metrics_comparison()` → Figure 7

```python
def plot_metrics_comparison(results, results_dir):
    metrics_names = ['Accuracy', 'AUC', 'Sensitivity', 'Specificity']
    
    for i, metric in enumerate(metrics_names):
        values = [results[m][metric] for m in models]
        ax.bar(x + offset, values, width, label=metric, color=colors[i])
```

Vẽ **grouped bar chart** để so sánh trực quan 4 chỉ số của 5 model cạnh nhau.

---

### `matplotlib.use('Agg')` — Backend Không Cần Màn Hình

```python
import matplotlib
matplotlib.use('Agg')
```

Dòng này quan trọng khi chạy trên **server hoặc script không có GUI**:
- Backend `Agg` render đồ thị vào buffer bộ nhớ và lưu file PNG
- Không cần cửa sổ hiển thị trên màn hình
- `plt.savefig()` lưu file; `plt.close()` giải phóng bộ nhớ

---

## Hàm `print_results_table()` — In Bảng So Sánh

```python
def print_results_table(results):
    print(f"{'Model':<20} {'Accuracy':>10} {'Sensitivity':>12} ...")
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['Accuracy']*100:>9.2f}% ...")
    
    best_model = max(results.items(), key=lambda x: x[1]['AUC'])
    print(f"\n🏆 Best Model: {best_model[0]} (AUC = {best_model[1]['AUC']:.4f})")
```

- `:<20` = căn trái, chiều rộng 20 ký tự
- `:>9.2f` = căn phải, 2 số thập phân
- `max(..., key=lambda x: x[1]['AUC'])` = tìm model có AUC cao nhất

---

## Sơ Đồ Luồng

```
y_test (nhãn thực)    predictions (dict)
        │                    │
        └────────┬───────────┘
                 │
         calculate_metrics()
                 │
         ┌───────┴──────────────────────────┐
         │  Accuracy, Sensitivity,          │
         │  Specificity, AUC                │
         │  fpr, tpr (cho ROC)              │
         │  TP, TN, FP, FN                  │
         └───────┬──────────────────────────┘
                 │
       ┌─────────┼──────────────────┐
       │         │                  │
 print_results   │           generate_all_plots()
  _table()       │                  │
  (console)      │    ┌─────────────┼──────────────┐
                 │    │             │              │
              results confusion   roc_curves  metrics_comparison
              dict    _matrices   .png        .png
                      .png        (Fig 6)     (Fig 7)
                      (Fig 5)
```

---

## Files Tạo Ra

```
results/
├── confusion_matrices.png   ← Figure 5 (grid 2×3 heatmap)
├── roc_curves.png           ← Figure 6 (5 đường cong + baseline)
└── metrics_comparison.png  ← Figure 7 (grouped bar chart)
```

---

*Trước: [03_explain_models.md](03_explain_models.md)*  
*Tiếp theo: [05_explain_explainability.md](05_explain_explainability.md)*
