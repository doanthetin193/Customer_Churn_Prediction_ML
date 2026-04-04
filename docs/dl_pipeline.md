# Tài liệu Pipeline Deep Learning (MLP)

**Học phần:** Học Máy và Ứng Dụng  
**Đề tài:** Dự đoán rời bỏ khách hàng trong ngành viễn thông  
**Nhánh:** `main_DL/` — phát triển độc lập, so sánh với nhánh ML truyền thống (`main/`)

---

## 1. Tổng quan pipeline

Pipeline DL nằm trong thư mục `main_DL/` và được điều phối bởi `main_DL/main.py`. Gồm 6 bước:

```
Load Data → Preprocess → Train/Val Split → Train MLP → Optimal Threshold → Evaluate + Save
```

**Điểm khác biệt chính so với ML:**
- Mô hình là **Multilayer Perceptron (MLP)** thay vì các thuật toán sklearn
- Có thêm bước **tách validation set** từ train set để monitor trong quá trình huấn luyện
- Có bước **tìm ngưỡng phân loại tối ưu** thay vì dùng cứng 0.5

---

## 2. Dữ liệu và tiền xử lý

Dữ liệu và tiền xử lý **hoàn toàn đồng bộ** với nhánh ML để so sánh công bằng:

| Bước | Xử lý |
|---|---|
| Lọc "Joined" | Giữ lại Churned + Stayed → **6.589 mẫu** |
| Missing values | Median (số) / Mode (phân loại) |
| Encoding | LabelEncoder cho 18 cột phân loại |
| Scaling | StandardScaler (fit on train only) |
| Split | Train 75% / Test 25%, stratify, random_state=42 |
| Imbalance | `class_weight='balanced'` — **không dùng SMOTE** |

### Chia thêm Validation Set

Sau bước split train/test, train set tiếp tục được chia thêm:

```
Train set (4.941 mẫu) → Train thực (80%) + Validation (20%)
                       = 3.953 mẫu      +  988 mẫu
```

Validation set dùng để:
- Theo dõi `val_auc` sau mỗi epoch
- Kích hoạt EarlyStopping và ReduceLROnPlateau
- Tìm ngưỡng phân loại tối ưu (Youden's J)

### Class Weights

```python
class_weight = {
    0: 0.698,   # Not Churned (majority) — trọng số thấp hơn
    1: 1.763    # Churned (minority) — trọng số cao hơn
}
```

Keras sẽ nhân loss của mẫu thuộc lớp 1 lên ~2.5× so với lớp 0, buộc model chú ý nhiều hơn đến khách hàng churn.

---

## 3. Kiến trúc mô hình MLP (`models_dl.py`)

### 3.1. Sơ đồ kiến trúc

```
Input (25 features)
    │
    ▼
Dense(256) → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.40)
    │
    ▼
Dense(128) → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.40)
    │
    ▼
Dense(64)  → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.20)
    │
    ▼
Dense(1) → Sigmoid
```

### 3.2. Hyperparameters

| Tham số | Giá trị | Lý do chọn |
|---|---|---|
| Hidden units | 256 → 128 → 64 | Đủ rộng để học phi tuyến, không quá sâu để tránh overfitting |
| Activation | **LeakyReLU (α=0.1)** | Tránh dying ReLU (neuron bị "chết" khi gradient = 0) |
| Batch Normalization | Có (sau mỗi Dense) | Ổn định gradient, tăng tốc hội tụ |
| Dropout | 0.4 / 0.4 / 0.2 | Regularization hiệu quả, block cuối nhẹ hơn |
| L2 regularization | λ = 3×10⁻⁴ | Phạt trọng số lớn, giảm overfitting |
| Optimizer | Adam | Adaptive learning rate, robustness cao |
| Learning rate | 5×10⁻⁴ | Hội tụ ổn định hơn 10⁻³ |
| Loss | BinaryCrossentropy | Chuẩn cho phân loại nhị phân |

### 3.3. Quá trình cải tiến kiến trúc

Phiên bản ban đầu (`main_DL/` gốc) dùng kiến trúc đơn giản hơn:

| Thành phần | Baseline | **Phiên bản cải tiến** |
|---|---|---|
| Depth | 3 blocks: 128→64→32 | 3 blocks: **256→128→64** |
| Activation | ReLU | **LeakyReLU(α=0.1)** |
| L2 regularization | Không | **λ = 3×10⁻⁴** |
| Dropout | 0.3 / 0.3 / 0.21 | **0.4 / 0.4 / 0.2** |
| Learning rate | 10⁻³ | **5×10⁻⁴** |
| Label smoothing | Không | Không (đã thử, không cải thiện) |

**Vấn đề phát hiện:** Phiên bản 4-block (256→128→64→32) có val_auc ~0.937 nhưng test_auc chỉ ~0.910 → khoảng cách 2.7% do overfitting. Nguyên nhân: số tham số (~43k) quá lớn so với chỉ ~3.953 mẫu train thực tế.  
**Giải pháp:** Thu gọn xuống 3 block + tăng dropout lên 0.4 + tăng L2 lên 3×10⁻⁴ → khoảng cách val/test giảm còn ~1.3%.

---

## 4. Huấn luyện (`train_mlp`)

### 4.1. Cấu hình training

| Tham số | Giá trị |
|---|---|
| Epochs tối đa | 150 |
| Batch size | 64 |
| Monitor metric | `val_auc` |

### 4.2. Callbacks

**EarlyStopping:**
```
monitor = val_auc (mode=max)
patience = 20 epochs
restore_best_weights = True
```
Nếu `val_auc` không tăng trong 20 epoch liên tiếp → dừng sớm và phục hồi trọng số tốt nhất.

**ReduceLROnPlateau:**
```
monitor = val_auc (mode=max)
factor = 0.5  (giảm LR đi một nửa)
patience = 5 epochs
min_lr = 5×10⁻⁶
```
Khi `val_auc` không cải thiện trong 5 epoch → giảm learning rate, giúp fine-tune gần tối ưu.

### 4.3. Kết quả training thực tế

- Early stopping kích hoạt tại **epoch 66**
- Best weights phục hồi từ **epoch 46**
- `val_auc` tốt nhất đạt: **~0.9438**

---

## 5. Tìm ngưỡng phân loại tối ưu (`find_optimal_threshold`)

Thay vì dùng ngưỡng mặc định 0.5, pipeline DL tìm ngưỡng tối ưu trên **validation set** bằng cách tối đa hoá **Youden's J statistic**:

$$J = \text{Sensitivity} + \text{Specificity} - 1$$

**Thuật toán:**
```
for threshold in [0.05, 0.06, ..., 0.94]:
    tính Sensitivity và Specificity tại ngưỡng đó
    tính J = Sensitivity + Specificity - 1
    lưu ngưỡng cho J tốt nhất
```

**Kết quả:** Ngưỡng tối ưu = **0.54** (thay vì 0.5 mặc định), Youden's J = 0.7472.

**Ý nghĩa:** Ngưỡng 0.54 có nghĩa là mô hình cần xác suất churn ≥ 54% mới phân loại là "Churned". Điều này cân bằng tốt hơn giữa Sensitivity và Specificity so với 0.5.

---

## 6. Kết quả đánh giá

### 6.1. Metrics trên test set

| Metric | Giá trị |
|---|---:|
| **Accuracy** | **81.74%** |
| **Sensitivity** | **84.37%** |
| **Specificity** | **80.69%** |
| **AUC** | **0.9132** |
| TP | 394 |
| TN | 953 |
| FP | 228 |
| FN | 73 |
| Threshold dùng | 0.54 |

### 6.2. Đầu ra hình ảnh (`main_DL/results/`)

| File | Mô tả |
|---|---|
| `confusion_matrix_dl.png` | Confusion matrix của MLP |
| `roc_curve_dl.png` | ROC curve của MLP |
| `training_history_dl.png` | Đồ thị Loss và AUC theo epoch (train vs val) |
| `metrics_dl.json` | Kết quả metrics dạng JSON |
| `mlp_churn_model.keras` | Model đã train, có thể load lại |

---

## 7. So sánh DL (MLP) vs ML (Random Forest — best model)

### 7.1. Bảng so sánh trực tiếp

| Metric | Random Forest (ML) | MLP (DL) | Chênh lệch |
|---|---:|---:|---|
| Accuracy | 86.65% | 81.74% | RF tốt hơn 4.91% |
| **Sensitivity** | 74.52% | **84.37%** | **DL tốt hơn 9.85%** ✓ |
| Specificity | 91.45% | 80.69% | RF tốt hơn 10.76% |
| AUC | 0.9264 | 0.9132 | RF tốt hơn 0.013 |

### 7.2. Phân tích từng metric

**Accuracy — RF thắng (+4.91%):**  
RF phân loại đúng nhiều mẫu hơn về tổng thể. Đây là do RF có Specificity rất cao (91.45%), tức là rất ít FP, nên Accuracy tổng thể cao.

**Sensitivity — DL thắng (+9.85%):**  
MLP phát hiện được **84.37%** số khách hàng thực sự sẽ rời bỏ (TP=394), trong khi RF chỉ phát hiện được **74.52%** (TP≈367). DL bỏ sót ít churner hơn (FN=73 vs ~120 của RF).

**Specificity — RF thắng (+10.76%):**  
RF ít cảnh báo nhầm (FP) hơn nhiều. MLP cảnh báo nhầm 228 khách không churn là sẽ churn, RF chỉ khoảng 130.

**AUC — RF thắng (0.013):**  
RF phân biệt churn/non-churn tốt hơn ở mọi ngưỡng. Tuy nhiên khoảng cách 0.0132 là nhỏ và cả hai đều đạt trên 0.91 — đều là mô hình tốt.

### 7.3. Tại sao Sensitivity của DL cao hơn RF?

Có 3 yếu tố kết hợp:

1. **Class weight mạnh hơn trong Keras:** Keras áp dụng class weight trực tiếp vào từng batch, loss của mẫu churn được nhân ×1.76 — ảnh hưởng trực tiếp hơn so với sklearn's `class_weight='balanced'` trong RF.

2. **Ngưỡng tối ưu (0.54):** Thay vì 0.5, việc dùng ngưỡng tối ưu theo Youden's J cân bằng tốt hơn Sensitivity và Specificity. RF dùng ngưỡng mặc định 0.5.

3. **Học phi tuyến sâu hơn:** MLP với 3 lớp ẩn có khả năng học các tương tác phức tạp giữa features mà cây quyết định hay ensemble không nắm được — đặc biệt có lợi cho việc phát hiện các pattern tinh tế của churner.

### 7.4. Khi nào nên chọn DL thay vì ML?

| Tiêu chí | Chọn ML (RF) | Chọn DL (MLP) |
|---|---|---|
| Ưu tiên Accuracy tổng thể | ✓ | |
| Ưu tiên Sensitivity (phát hiện churner) | | ✓ |
| Chi phí FN cao (bỏ sót churner tốn kém) | | ✓ |
| Cần giải thích mô hình (SHAP/LIME) | ✓ | |
| Dataset lớn (>50k mẫu) | | ✓ |
| Cần tốc độ inference | ✓ | |

**Trong bài toán churn của viễn thông:** Chi phí thu hút khách hàng mới cao gấp 5–10× chi phí giữ chân. Bỏ sót 1 churner (FN) tốn kém hơn nhiều so với cảnh báo nhầm 1 non-churner (FP). Vì vậy **Sensitivity cao của DL là lợi thế thực tế quan trọng**.

---

## 8. Cách chạy

```bash
# Từ thư mục gốc của project
python main_DL/main.py
```

Hoặc với virtual environment:

```bash
d:\Customer_Churn_Prediction_ML\.venv\Scripts\python.exe main_DL/main.py
```

---

## 9. Cấu trúc thư mục `main_DL/`

```
main_DL/
├── main.py               # Entry point, điều phối toàn bộ pipeline
├── data_loader.py        # Load CSV, thống kê phân phối nhãn
├── data_preprocessing.py # Filter Joined, encode, scale, split (đồng bộ ML)
├── models_dl.py          # Kiến trúc MLP, train, optimal threshold
├── evaluation.py         # Tính metrics, vẽ biểu đồ, lưu JSON
├── requirements.txt      # tensorflow-cpu>=2.12.0 + các deps
└── results/              # Đầu ra (tự sinh khi chạy)
    ├── confusion_matrix_dl.png
    ├── roc_curve_dl.png
    ├── training_history_dl.png
    ├── metrics_dl.json
    └── mlp_churn_model.keras
```

---

## 10. Tóm tắt các thay đổi so với DL baseline

| # | Thay đổi | Baseline | Phiên bản cải tiến | Tác động |
|---|---|---|---|---|
| 1 | Filter "Joined" | Không lọc | Lọc còn 6.589 mẫu | Dữ liệu nhất quán với ML |
| 2 | SMOTE | Bật | Tắt + `class_weight='balanced'` | Không tạo dữ liệu giả |
| 3 | Kiến trúc | 128→64→32 (3 block) | **256→128→64 (3 block)** | Wider, học tốt hơn |
| 4 | Activation | ReLU | **LeakyReLU(α=0.1)** | Tránh dying neuron |
| 5 | L2 regularization | Không | **λ = 3×10⁻⁴** | Giảm overfitting |
| 6 | Dropout | 0.3 / 0.3 / 0.21 | **0.4 / 0.4 / 0.2** | Giảm val/test gap |
| 7 | Learning rate | 10⁻³ | **5×10⁻⁴** | Hội tụ ổn định hơn |
| 8 | Threshold | Cứng 0.5 | **Optimal Youden's J = 0.54** | Sensitivity tăng |
| 9 | Early stopping patience | 15 | **20** | Cho model hội tụ đủ |
| — | **Sensitivity (kết quả)** | ~79.23% | **84.37%** | **+5.14%** |
| — | **AUC (kết quả)** | ~0.9066 | **0.9132** | **+0.007** |
