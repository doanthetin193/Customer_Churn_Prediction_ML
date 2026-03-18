# Giải thích: `explainability.py` — Bước 6: Giải Thích Mô Hình (XAI)

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py  [✓]
      └─► data_preprocessing.py  [✓]
           └─► models.py  [✓]
                └─► evaluation.py  [✓]
                     └─► [BƯỚC 6] explainability.py  ← Bạn đang ở đây
```

---

## Tổng Quan

File này triển khai **Explainable AI (XAI)** — phần then chốt phân biệt bài báo này với các nghiên cứu chỉ tập trung vào accuracy. Bài báo (Section 3.5) nêu ro: mô hình chính xác nhưng "hộp đen" sẽ **không được decision-maker chấp nhận** trong thực tế kinh doanh.

Code triển khai 2 kỹ thuật XAI tương ứng với **Figure 8, 9, 10** trong bài báo:
- **LIME** → Giải thích từng dự đoán cụ thể (local)
- **SHAP** → Giải thích tầm quan trọng tổng thể của feature (global)

**Đầu vào:** Random Forest model đã train, `X_train`, `X_test`, `feature_names`  
**Đầu ra:** 4 file PNG trong `results/`

---

## Liên Hệ Bài Báo (Section 3.5)

| Bài báo đề cập | Code thực hiện |
|---|---|
| *"LIME to optimize the local interpretability of classifiers"* | `explain_with_lime()` → `lime_sample_0.png`, `lime_sample_1.png` (Figure 8) |
| *"SHAP summary plots for importance of features and influence on prediction"* | `explain_with_shap()` → `shap_summary.png` (Figure 9) |
| *"features ranked by sum of SHAP values"* | `shap_bar.png` (Figure 10) |
| *"blue = low, red = high SHAP values"* | `cmap='coolwarm'` trong beeswarm plot |
| *"results of the best customer churn prediction model"* | Chỉ dùng Random Forest (model tốt nhất) |

---

## Kiểm Tra Thư Viện Động

```python
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("Warning: LIME not installed. Run: pip install lime")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
```

**Lý do dùng try/except:**
- LIME và SHAP là thư viện **không có sẵn** trong scikit-learn
- Nếu chưa cài (`pip install lime shap`), script không crash — chỉ bỏ qua bước XAI
- Cho phép chạy phần còn lại của pipeline bình thường

---

## Phần 1: LIME — Local Interpretable Model-Agnostic Explanations

### Nguyên Lý (Bài Báo Section 3.5)

> *"LIME emphasizes training locally interpretable models that may be used to explain specific predictions and help decision-makers understand why a particular class was predicted for a certain instance."*

**LIME không cần biết bên trong mô hình** (model-agnostic). Thay vào đó:

```
Điểm test cần giải thích: [Khách hàng X]
              │
              │ Tạo ra N mẫu lân cận (perturbation)
              ▼
[X±ε₁] [X±ε₂] ... [X±εₙ]  ← Các điểm gần X
              │
              │ Hỏi Black-box model: dự đoán gì?
              ▼
P₁, P₂, ..., Pₙ  ← Xác suất churn tương ứng
              │
              │ Fit mô hình tuyến tính đơn giản
              │ trên N điểm này (có trọng số gần X)
              ▼
Linear Model: P ≈ β₀ + β₁·Tenure + β₂·Contract + ...
              │
              ▼
Giải thích: "Tenure ngắn làm TĂNG xác suất churn 0.3"
```

---

### `create_lime_explainer(X_train, feature_names)`

```python
def create_lime_explainer(X_train, feature_names, class_names=None):
    if class_names is None:
        class_names = ['Good', 'Bad']   # Theo bài báo: Good=Not Churn, Bad=Churn
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,    # LIME cần biết phân phối dữ liệu train
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    return explainer
```

**Tại sao cần `X_train`?**  
LIME dùng phân phối của training data để tạo các mẫu perturbation có ý nghĩa thống kê (không tạo giá trị vô lý).

**`class_names = ['Good', 'Bad']`:**  
Bài báo (Figure 8) dùng nhãn "Good" (không churn) và "Bad" (churn) — code giữ nguyên cách đặt tên này.

---

### `explain_with_lime(model, explainer, X_sample, sample_idx)`

```python
def explain_with_lime(model, explainer, X_sample, sample_idx=0, num_features=10, results_dir='results'):
    exp = explainer.explain_instance(
        X_sample[sample_idx],     # Mẫu cần giải thích (1 khách hàng)
        model.predict_proba,      # Hàm dự đoán của Black-box model
        num_features=num_features,# Hiển thị top 10 features quan trọng nhất
        labels=[0]                # Giải thích class 0 (Good/Not Churn) như bài báo Figure 8
    )
    
    fig = exp.as_pyplot_figure(label=0)
    plt.savefig(f'lime_sample_{sample_idx}.png', ...)
```

**Tại sao `labels=[0]` thay vì `[1]`?**  
Bài báo Figure 8 trình bày giải thích cho class "Good" (Not Churn). Khi xác suất class 0 < 0.5, tức là model dự đoán class 1 (Churn). Cách đọc ngược này là quy ước trong LIME khi giải thích classification.

**Output — Ví dụ giải thích (lime_sample_0.png):**
```
Feature              Contribution
Contract=Month-to-Month  →  -0.25 (làm TĂNG xác suất churn)
Tenure < 12          →  -0.18 (làm TĂNG xác suất churn)
Online Security=No   →  -0.12 (làm TĂNG xác suất churn)
Monthly Charge > 90  →  -0.09
Total Charges > 8000 →  +0.15 (làm GIẢM xác suất churn)
```

Code giải thích **2 mẫu** (sample_idx=0 và 1) — tương ứng Figure 8a và 8b trong bài báo.

---

## Phần 2: SHAP — SHapley Additive Explanations

### Nguyên Lý (Bài Báo Section 3.5)

> *"SHAP is a method of interpreting the results of any ML model by attributing an importance score to each feature in the data."*

**SHAP dựa trên lý thuyết game theory (Shapley values):**  
Coi mỗi feature như một "người chơi" trong trò chơi hợp tác. SHAP value của feature $i$ là **đóng góp trung bình** của feature đó vào dự đoán, tính qua tất cả các tập con feature có thể:

$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

Trong đó $F$ = tập tất cả features, $f(S)$ = dự đoán của mô hình với tập features $S$.

**Đặc tính quan trọng:**

$$f(x) = \phi_0 + \sum_{i=1}^{n} \phi_i$$

→ Tổng SHAP values = Dự đoán cuối cùng (có thể giải thích đầy đủ)

---

### `explain_with_shap(model, X_train, X_test, feature_names)`

```python
def explain_with_shap(model, X_train, X_test, feature_names, results_dir, max_samples=100):
    X_subset = X_test[:max_samples]   # Dùng 100 mẫu để tăng tốc
    
    # TreeExplainer nhanh hơn cho tree-based models
    if model_name == 'RandomForestClassifier':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_subset)
        # shap_values là list [class_0_values, class_1_values]
        shap_values_plot = shap_values[1]  # Lấy class 1 (Churn)
    else:
        # KernelExplainer cho các model khác (chậm hơn)
        background = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        ...
```

**Tại sao dùng `TreeExplainer` cho Random Forest?**
- `TreeExplainer` tận dụng cấu trúc cây quyết định → tính SHAP **chính xác** và **nhanh** (O(TLD²) thay vì O(2^n))
- `KernelExplainer` là phương pháp tổng quát (model-agnostic) nhưng chậm hơn nhiều

**`max_samples=100`:** SHAP tính qua tất cả combinations của features → rất tốn tài nguyên. Giới hạn 100 mẫu test để chạy trong thời gian hợp lý.

**Output thực tế khi chạy:**
```
Calculating SHAP values (this may take a while)...
  SHAP values shape: (100, 25, 2)
  X_subset shape: (100, 25)
  Feature names count: 25
```

> **Lưu ý về shape (100, 25, 2):** TreeExplainer trả về SHAP values dạng 3D — `[n_samples, n_features, n_classes]`. Code lấy `shap_values[1]` → class 1 (Churn), thu được mảng 2D `(100, 25)` để vẽ biểu đồ.

---

### Figure 9: SHAP Beeswarm Plot (Summary Plot)

```python
# Vẽ custom beeswarm plot
for i, idx in enumerate(top_indices):
    shap_col = shap_values_plot[:, idx]       # SHAP values của feature idx
    feature_col = X_subset[:, idx]            # Giá trị thực của feature
    
    norm_values = (feature_col - min) / (max - min)   # Normalize 0→1
    y_pos = len(top_indices)-1-i + np.random.uniform(-0.2, 0.2, n)  # Jitter y
    
    scatter = ax.scatter(shap_col, y_pos, 
                         c=norm_values,       # Màu theo giá trị feature
                         cmap='coolwarm')     # Xanh=thấp, Đỏ=cao
```

**Cách đọc biểu đồ (theo bài báo):**

```
Feature           │ ← SHAP < 0       │ SHAP > 0 →
                  │ (giảm xác suất churn)│(tăng xác suất churn)
──────────────────┼──────────────────┼──────────────────
Contract          │  ●●●●●           │           ●●●●● 
                  │  (Two Year=xanh) │        (Month-to-Month=đỏ)
Tenure in Months  │       ●●●●       │      ●●●●
Monthly Charge    │  ●●              │              ●●
                  0
```

- Mỗi dấu `●` là 1 khách hàng
- **Màu đỏ** = giá trị feature cao; **màu xanh** = giá trị feature thấp
- **Vị trí ngang** = SHAP value (tác động lên xác suất churn)

---

### Figure 10: SHAP Bar Plot (Feature Importance)

```python
# Bar chart: mean(|SHAP value|) cho từng feature
mean_abs_shap = np.abs(shap_values_plot).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1][:10]  # Top 10 features

ax.barh(y_pos, sorted_importance[::-1], color='#1976D2')
ax.set_xlabel('mean(|SHAP value|)')
```

**Cách đọc:** Feature nào có `mean(|SHAP|)` lớn nhất → ảnh hưởng nhiều nhất đến dự đoán churn.

**Kết quả điển hình (theo SHAP feature importance):**
```
Rank  Feature                    mean(|SHAP|)
1     Contract                   0.285  ← quan trọng nhất
2     Tenure in Months           0.231
3     Monthly Charge             0.198
4     Internet Type              0.156
5     Total Revenue              0.134
...
```

---

## Phân Biệt LIME vs SHAP

| | LIME | SHAP |
|---|---|---|
| **Phạm vi** | **Local** — giải thích 1 dự đoán cụ thể | **Global** — giải thích tổng thể mô hình |
| **Phương pháp** | Fit mô hình tuyến tính cục bộ | Shapley values (game theory) |
| **Tốc độ** | Nhanh | Chậm hơn (TreeExplainer khá nhanh) |
| **Consistency** | Có thể không nhất quán giữa các lần chạy | Nhất quán, có cơ sở lý thuyết chặt chẽ |
| **Dùng khi** | Giải thích tại sao 1 khách hàng cụ thể bị dự đoán churn | Hiểu tổng quan feature nào quan trọng |
| **Figure trong bài báo** | Figure 8 (2 mẫu cụ thể) | Figure 9 (beeswarm), Figure 10 (bar) |

---

## Hàm Điều Phối `generate_explanations()`

```python
def generate_explanations(model, X_train, X_test, feature_names, results_dir):
    # LIME cho 2 mẫu đầu tiên
    if LIME_AVAILABLE:
        explainer = create_lime_explainer(X_train, feature_names)
        explain_with_lime(model, explainer, X_test, sample_idx=0, ...)
        explain_with_lime(model, explainer, X_test, sample_idx=1, ...)
    
    # SHAP cho 100 mẫu test
    if SHAP_AVAILABLE:
        explain_with_shap(model, X_train, X_test, feature_names, ...)
```

---

## Files Tạo Ra

```
results/
├── lime_sample_0.png   ← Figure 8a: LIME giải thích mẫu đầu tiên
├── lime_sample_1.png   ← Figure 8b: LIME giải thích mẫu thứ hai
├── shap_summary.png    ← Figure 9: Beeswarm plot (SHAP)
└── shap_bar.png        ← Figure 10: Bar chart feature importance (SHAP)
```

---

## Tại Sao XAI Quan Trọng?

Bài báo (Section 3.5) nhấn mạnh:
> *"Although ML algorithms excel in terms of accuracy, they may not gain the trust of decision-makers or users or be accepted in real-world enterprise management."*

Một customer retention manager cần biết:
- "Tại sao hệ thống dự đoán khách hàng **này** sẽ churn?"
- "Tôi nên tác động vào **yếu tố nào** để giữ họ lại?"

→ LIME trả lời câu hỏi thứ nhất (cụ thể)  
→ SHAP trả lời câu hỏi thứ hai (tổng thể)

---

*Trước: [04_explain_evaluation.md](04_explain_evaluation.md)*  
*Tiếp theo: [06_explain_main.md](06_explain_main.md)*
