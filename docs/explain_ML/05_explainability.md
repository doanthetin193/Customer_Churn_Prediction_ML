# Giải thích: `explainability.py` — Bước 5: Giải Thích Mô Hình (XAI)

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py (BƯỚC 1)
      └─► data_preprocessing.py (BƯỚC 2)
           └─► models.py (BƯỚC 3)
                └─► evaluation.py (BƯỚC 4)
                     └─► [BƯỚC 5] explainability.py  ← Bạn đang ở đây
```

---

## Mục đích

Random Forest là **black box** — không thể biết tại sao nó ra quyết định đó. File này dùng 2 kỹ thuật XAI (Explainable AI) để mở hộp đen đó:

- **LIME** — giải thích _cục bộ_ (local): tại sao model dự đoán MÃU này là Churn?
- **SHAP** — giải thích _toàn cục_ (global): feature nào quan trọng nhất cho TẤT CẢ dự đoán?

---

## Liên hệ với Bài Báo (Section 3.5 — XAI)

> *"LIME is used to explain individual predictions by approximating the model locally with a simpler interpretable model. SHAP uses a game theory approach to fairly distribute the contribution of each feature."*

| Nội dung bài báo | Tương ứng trong code |
|---|---|
| Figure 8 — LIME local explanation (class "Good") | `explain_with_lime(..., labels=[0])` |
| Figure 9 — SHAP beeswarm/summary plot | `shap_summary.png` |
| Figure 10 — SHAP bar plot feature importance | `shap_bar.png` |
| Model được giải thích: Random Forest | `rf_model = trained_models['Random Forest']` |
| Class names: 'Good' (Not Churned), 'Bad' (Churned) | `class_names=['Good', 'Bad']` |

---

## Pattern Import an toàn (Try/Except)

```python
# LIME
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Run: pip install lime")

# SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Run: pip install shap")
```

**Tại sao dùng try/except thay vì import trực tiếp?**

LIME và SHAP là thư viện tuỳ chọn (optional dependencies) — không nằm trong stdlib Python. Nếu import trực tiếp và user chưa cài:

```python
import lime  # → ModuleNotFoundError: No module named 'lime'
```

Toàn bộ file crash, không chạy được gì. Với try/except, code vẫn chạy bình thường, chỉ bỏ qua phần XAI và in warning thay vì crash.

`LIME_AVAILABLE` và `SHAP_AVAILABLE` là **feature flag** — các hàm sau kiểm tra flag này trước khi chạy.

---

## Phân tích Code Chi Tiết

### Hàm 1: `create_lime_explainer(X_train, feature_names, class_names)`

```python
def create_lime_explainer(X_train, feature_names, class_names=None):
    if not LIME_AVAILABLE:
        return None

    if class_names is None:
        class_names = ['Good', 'Bad']  # Khớp với Figure 8 bài báo

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer
```

**`LimeTabularExplainer` cần gì?**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `training_data` | `X_train` (scaled) | LIME cần phân phối train để sample perturbations xung quanh mẫu |
| `feature_names` | 25 tên feature | Để hiển thị tên trong plot thay vì chỉ số |
| `class_names` | `['Good', 'Bad']` | Tên class. 'Good' = class 0 (Stayed), 'Bad' = class 1 (Churned) |
| `mode='classification'` | classification | Phân biệt với regression |

**Tại sao `class_names=['Good', 'Bad']` khớp với bài báo?**

Bài báo Figure 8 label class 0 là 'Good' và class 1 là 'Bad'. Đây chỉ là tên hiển thị — không ảnh hưởng logic model.

---

### Hàm 2: `explain_with_lime(model, explainer, X_sample, sample_idx, ...)`

```python
def explain_with_lime(model, explainer, X_sample, sample_idx=0, num_features=10, results_dir='results'):
    if not LIME_AVAILABLE or explainer is None:
        return None

    # Giải thích mẫu tại index sample_idx
    exp = explainer.explain_instance(
        X_sample[sample_idx],       # Mẫu cần giải thích
        model.predict_proba,        # Hàm predict của model
        num_features=num_features,  # Hiển thị top 10 features
        labels=[0]                  # Giải thích class 0 (Good) như Figure 8
    )

    fig = exp.as_pyplot_figure(label=0)
    fig.suptitle(f'LIME Explanation - Sample {sample_idx} (Figure 8)')
    plt.savefig(os.path.join(results_dir, f'lime_sample_{sample_idx}.png'), ...)
```

**LIME hoạt động như thế nào?**

1. Lấy mẫu cần giải thích: $x_0$ (một khách hàng cụ thể)
2. Tạo nhiều mẫu perturbation xung quanh $x_0$ (thay đổi nhỏ từng feature)
3. Cho Random Forest predict trên tất cả mẫu perturbation
4. Fit một **Linear Regression đơn giản** (interpretable) trên các perturbation đó
5. Hệ số của Linear Regression = giải thích LIME → feature nào đẩy prediction lên/xuống

**`labels=[0]` — tại sao giải thích class 0 (Good) chứ không phải class 1 (Bad)?**

Bài báo Figure 8 hiển thị explanation cho class 'Good'. Trong bài toán churn, khách hàng **không churn** (Good) là nhóm ta muốn hiểu: tại sao model nghĩ người này sẽ ở lại?

**`num_features=10`:** SHAP decompose tất cả 25 features nhưng LIME plot chỉ top 10 có ảnh hưởng lớn nhất để plot gọn.

**Trong `main.py`, LIME được gọi 2 lần:**

```python
explain_with_lime(model, explainer, X_test, sample_idx=0)  → lime_sample_0.png
explain_with_lime(model, explainer, X_test, sample_idx=1)  → lime_sample_1.png
```

2 mẫu: một Stayed, một Churned → so sánh để thấy sự khác biệt.

---

### Hàm 3: `explain_with_shap(model, X_train, X_test, feature_names, ...)`

```python
def explain_with_shap(model, X_train, X_test, feature_names, results_dir='results', max_samples=100):
    X_subset = X_test[:max_samples]  # Giới hạn 100 mẫu cho tốc độ

    model_name = type(model).__name__

    if model_name == 'RandomForestClassifier':
        explainer = shap.TreeExplainer(model)  # Nhanh, chính xác cho tree-based models
        shap_values = explainer.shap_values(X_subset)
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1]  # Lấy class 1 (Churn)
    else:
        background = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_subset)
        shap_values_plot = shap_values[1]
```

**Tại sao dùng `TreeExplainer` cho Random Forest?**

| Explainer | Phù hợp với | Tốc độ | Độ chính xác |
|---|---|---|---|
| `TreeExplainer` | Tree-based (RF, DT, XGBoost) | Rất nhanh | Chính xác (exact Shapley) |
| `KernelExplainer` | Bất kỳ model nào | Chậm | Xấp xỉ (approximate) |

Random Forest là tree-based → `TreeExplainer` cho kết quả chính xác hơn và nhanh hơn nhiều.

**`shap_values[1]` — lấy SHAP values của class 1 (Churn):**

`TreeExplainer` với binary classification trả về `list` gồm 2 phần tử: SHAP values cho class 0 và class 1. Lấy `[1]` để phân tích feature importance cho việc predict Churn.

**`max_samples=100`:** SHAP tính O(n × features × trees) → chậm với toàn bộ test set 1648 mẫu. Dùng 100 mẫu đại diện là trade-off tốt giữa tốc độ và accuracy giải thích.

---

**Figure 9 — SHAP Beeswarm Plot (custom implementation):**

```python
for i, idx in enumerate(top_indices):
    shap_col    = np.asarray(shap_values_plot[:, idx]).flatten()
    feature_col = np.asarray(X_subset[:, idx]).flatten()

    # Normalize feature values 0→1 for color mapping
    norm_values = (feature_col - feature_col.min()) / (feature_col.max() - feature_col.min())

    # Y jitter để các điểm không chồng lên nhau
    y_pos = len(top_indices) - 1 - i + np.random.uniform(-0.2, 0.2, n_samples)

    scatter = ax.scatter(shap_col, y_pos, c=norm_values, cmap='coolwarm', alpha=0.7, s=20)
```

**Cách đọc beeswarm plot:**

- Mỗi điểm = 1 mẫu (1 khách hàng)
- Trục X = SHAP value: dương (+) → đẩy prediction về Churn; âm (-) → đẩy về Stayed
- Màu: đỏ = feature value cao, xanh = feature value thấp
- Y axis = tên feature, sắp xếp theo mean |SHAP| giảm dần

Ví dụ: Feature "Tenure in Months" có SHAP âm lớn với màu xanh (tenure thấp) → khách mới gia nhập có nhiều khả năng Churn.

---

**Figure 10 — SHAP Bar Plot:**

```python
mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:n_top]

ax.barh([feature_names[i] for i in top_indices[::-1]],
        mean_abs_shap[top_indices[::-1]], ...)
```

**Mean absolute SHAP:**

$$\text{Feature Importance}_j = \frac{1}{n} \sum_{i=1}^{n} |\phi_j(x_i)|$$

Trung bình giá trị tuyệt đối SHAP trên tất cả mẫu → feature nào có giá trị lớn nhất là quan trọng nhất về mặt tổng thể.

`np.argsort()[::-1]` — sắp xếp tăng dần rồi đảo ngược → giảm dần.

---

### Hàm 4: `generate_explanations(model, X_train, X_test, feature_names, results_dir)`

```python
def generate_explanations(model, X_train, X_test, feature_names, results_dir='results'):
    if not (LIME_AVAILABLE or SHAP_AVAILABLE):
        print("Neither LIME nor SHAP available. Skipping XAI.")
        return

    print("Generating XAI Explanations for Random Forest...")

    explainer = create_lime_explainer(X_train, feature_names)

    if LIME_AVAILABLE and explainer:
        explain_with_lime(model, explainer, X_test, sample_idx=0, results_dir=results_dir)
        explain_with_lime(model, explainer, X_test, sample_idx=1, results_dir=results_dir)

    if SHAP_AVAILABLE:
        explain_with_shap(model, X_train, X_test, feature_names, results_dir)
```

Hàm wrapper này được gọi từ `main.py`:

```python
rf_model = trained_models['Random Forest']
generate_explanations(
    rf_model,
    data['X_train'], data['X_test'],
    data['feature_names'],
    results_dir
)
```

---

## Files output được tạo ra

| File | Nội dung | Tương ứng với |
|---|---|---|
| `results/lime_sample_0.png` | LIME explanation cho mẫu test[0] | Figure 8 bài báo |
| `results/lime_sample_1.png` | LIME explanation cho mẫu test[1] | Figure 8 (mẫu khác) |
| `results/shap_summary.png` | SHAP beeswarm: phân phối feature importance | Figure 9 bài báo |
| `results/shap_bar.png` | SHAP bar: mean |SHAP| từng feature | Figure 10 bài báo |

---

## So sánh LIME vs SHAP

| Tiêu chí | LIME | SHAP |
|---|---|---|
| **Phạm vi giải thích** | Cục bộ (1 mẫu) | Toàn cục (tất cả mẫu) |
| **Nền tảng toán học** | Linear surrogate model | Shapley values (game theory) |
| **Tốc độ** | Nhanh | Chậm hơn (nhưng TreeExplainer bù lại) |
| **Độ chính xác** | Xấp xỉ cục bộ | Chính xác (với TreeExplainer) |
| **Ứng dụng** | Giải thích với 1 khách hàng cụ thể | Phân tích feature quan trọng toàn bộ |
| **Output** | Bar chart features → class | Beeswarm + Bar chart |

**Lý do bài báo dùng cả hai:**

LIME và SHAP bổ sung cho nhau — LIME nói "tại sao *người này* bị dự đoán là Churn", còn SHAP nói "feature nào *nhìn chung* quan trọng nhất để phân biệt Churn vs Stayed" với toàn bộ dataset.
