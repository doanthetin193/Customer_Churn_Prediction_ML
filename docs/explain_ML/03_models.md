# Giải thích: `models.py` — Bước 3: Xây Dựng và Huấn Luyện Mô Hình

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py (BƯỚC 1)
      └─► data_preprocessing.py (BƯỚC 2)
           └─► [BƯỚC 3] models.py  ← Bạn đang ở đây
                └─► evaluation.py
                     └─► explainability.py
```

---

## Mục đích

Định nghĩa, cấu hình và huấn luyện **5 mô hình phân loại** theo đúng bài báo (Section 3.3). Mỗi model được chọn vì sự đa dạng về nguyên lý hoạt động để so sánh công bằng.

---

## Liên hệ với Bài Báo (Section 3.3 — Methodology)

> *"Five machine learning algorithms are selected: LR, KNN, NB, DT, and RF. These algorithms were selected because they process data differently and are effective in performing classification tasks."*

| Thuật toán trong bài báo | Tên trong code | Class sklearn |
|---|---|---|
| Logistic Regression (LR) | `'Logistic Regression'` | `LogisticRegression` |
| K-Nearest Neighbor (KNN) | `'KNN'` | `KNeighborsClassifier` |
| Naïve Bayes (NB) | `'Naive Bayes'` | `GaussianNB` |
| Decision Tree (DT) | `'Decision Tree'` | `DecisionTreeClassifier` |
| Random Forest (RF) | `'Random Forest'` | `RandomForestClassifier` |

---

## Phân tích Code Chi Tiết

### Hàm 1: `get_models()` — Định nghĩa 5 model

```python
def get_models() -> dict:
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance'
        ),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
    }
    return models
```

---

#### Model 1: Logistic Regression

**Nguyên lý:** Dùng hàm sigmoid để ước tính xác suất thuộc class 1 (Churn):

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

Nếu $P > 0.5$ → dự đoán Churn, ngược lại → Stayed.

**Tham số quan trọng:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `max_iter=1000` | 1000 | Default=100 không đủ cho 25 features → tăng để convergence |
| `class_weight='balanced'` | tự động | Tăng weight cho class thiểu số (Churn 28%) |
| `random_state=42` | 42 | Reproducibility |

**`class_weight='balanced'` tính như thế nào?**

$$w_i = \frac{n\_samples}{n\_classes \times n\_samples\_i}$$

$$w_{Churn} = \frac{6589}{2 \times 1869} \approx 1.763$$
$$w_{Stayed} = \frac{6589}{2 \times 4720} \approx 0.698$$

Nghĩa là mỗi lần model sai với khách Churn, mất mát gấp 2.5× so với sai với khách Stayed.

---

#### Model 2: KNN (K-Nearest Neighbors)

**Nguyên lý:** Không có bước training thực sự — khi predict, tìm K điểm gần nhất trong train set rồi vote.

$$\hat{y} = \arg\max_c \sum_{i \in N_K(x)} w_i \cdot \mathbb{1}[y_i = c]$$

**Tham số quan trọng:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `n_neighbors=5` | 5 | Ba báo K=5 là default phổ biến |
| `weights='distance'` | distance | Điểm gần hơn có trọng số lớn hơn: $w = 1/d$ |

**Tại sao `weights='distance'` thay vì `'uniform'`?**

`'uniform'`: mỗi hàng xóm có quyền vote bằng nhau → hàng xóm gần và xa có ảnh hưởng như nhau.
`'distance'`: hàng xóm gần ảnh hưởng nhiều hơn → prediction chính xác hơn trong không gian feature phức tạp.

**KNN không có `class_weight`** — model này tự xử lý imbalance qua `distance weighting`.

---

#### Model 3: Naïve Bayes (Gaussian)

**Nguyên lý:** Dùng định lý Bayes với giả định conditional independence giữa features:

$$P(y|x) \propto P(y) \prod_{i=1}^n P(x_i|y)$$

Với Gaussian NB, $P(x_i|y) \sim \mathcal{N}(\mu_{iy}, \sigma_{iy}^2)$.

```python
'Naive Bayes': GaussianNB()
```

**GaussianNB không có `class_weight`** — model tự ước tính prior $P(y)$ từ tần suất trong training data. Với dữ liệu mất cân bằng 72/28, prior sẽ thiên về Stayed → model có xu hướng ít predict Churn hơn.

Đây là mô hình **đơn giản nhất** trong 5 — không có hyperparameter để tune, training cực nhanh, nhưng giả định Gaussian và independence thường không đúng với thực tế.

---

#### Model 4: Decision Tree

**Nguyên lý:** Chia dữ liệu thành các nút (node) theo ngưỡng feature để minimize impurity (Gini):

$$Gini = 1 - \sum_{k} p_k^2$$

Tại mỗi node, chọn feature và threshold có Information Gain lớn nhất.

| Tham số | Giá trị | Lý do |
|---|---|---|
| `class_weight='balanced'` | balanced | Xử lý imbalanced data |
| `random_state=42` | 42 | Đảm bảo cấu trúc cây giống nhau mỗi lần chạy |
| `max_depth` | None | Không giới hạn — cây có thể grow đầy đủ (dễ overfit nhưng đây là baseline) |

Decision Tree **không tuning max_depth** → có thể overfit, nhưng đây là cách bài báo dùng cho baseline comparison.

---

#### Model 5: Random Forest (Best Model)

**Nguyên lý:** Ensemble của nhiều Decision Tree độc lập, mỗi cây train trên bootstrap sample và chọn ngẫu nhiên subset features → giảm variance, tránh overfit.

$$\hat{y} = \text{majority\_vote}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_{200})$$

**Tham số chi tiết:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `n_estimators=200` | 200 cây | Đủ nhiều để stable, bài báo dùng 200 |
| `max_depth=15` | 15 | Kiểm soát depth → giảm overfit |
| `min_samples_split=5` | 5 | Node cần ≥5 mẫu mới được split |
| `min_samples_leaf=2` | 2 | Leaf cần ≥2 mẫu → leaf ít noisy hơn |
| `max_features='sqrt'` | $\sqrt{25} \approx 5$ | Randomness: mỗi split chỉ xem xét 5 features |
| `class_weight='balanced'` | balanced | Xử lý imbalance 72/28 |
| `bootstrap=True` | True | Sampling with replacement cho diversity |
| `random_state=42` | 42 | Reproducibility |
| `n_jobs=-1` | -1 | Dùng tất cả CPU cores → training nhanh hơn |

**Kết quả thực tế (RF là model tốt nhất):**

| Metric | RF | Paper (Chang et al. 2024) |
|---|---|---|
| Accuracy | **86.65%** | 86.94% |
| Sensitivity | 74.52% | 85.47% |
| Specificity | 91.45% | 88.39% |
| AUC | **0.9264** | 0.95 |

---

### Hàm 2: `get_tuned_random_forest(X_train, y_train)` — GridSearch (Không dùng)

```python
def get_tuned_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
```

Hàm này **tồn tại trong code nhưng KHÔNG được gọi** (`tune_rf=False` trong `main.py`).

**Lý do không dùng GridSearch:**
- Bài báo không đề cập GridSearchCV
- Đã hardcode hyperparameter tốt vào `get_models()` → không cần tuning
- GridSearch với `param_grid` trên mất ~30 phút → chậm không cần thiết
- $2 \times 3 \times 2 \times 2 = 24$ combinations × 5-fold CV = 120 model fits

---

### Hàm 3 & 4: `train_model()` và `train_all_models()`

```python
def train_all_models(X_train, y_train, tune_rf=True) -> dict:
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        if name == 'Random Forest' and tune_rf:
            trained_model = get_tuned_random_forest(X_train, y_train)
        else:
            trained_model = train_model(model, X_train, y_train)
        trained_models[name] = trained_model

    return trained_models
```

`train_model()` chỉ là wrapper cho `.fit()`:

```python
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
```

**Quá trình train thực tế (khi `tune_rf=False`):**

| Model | Thời gian train (ước tính) | Số tham số học |
|---|---|---|
| Logistic Regression | ~1 giây | 26 (25 weights + bias) |
| KNN | ~0 giây | 0 (lazy learner) |
| Naive Bayes | ~0.1 giây | 50 (μ, σ cho mỗi feature × 2 class) |
| Decision Tree | ~1 giây | Cấu trúc cây |
| Random Forest | ~15 giây | 200 cây |

---

### Hàm 5 & 6: `predict()` và `predict_all_models()`

```python
def predict(model, X):
    y_pred = model.predict(X)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]  # Xác suất class 1 (Churn)
    else:
        y_prob = y_pred  # Fallback (GaussianNB có predict_proba nên không đi nhánh này)
    return y_pred, y_prob
```

**`predict_proba(X)[:, 1]` — lấy cột index 1:**

`predict_proba()` trả về `(n_samples, 2)` — cột 0 là xác suất class 0 (Stayed), cột 1 là xác suất class 1 (Churn). Lấy `[:, 1]` để dùng cho AUC.

---

## So sánh 5 thuật toán

| Tiêu chí | LR | KNN | NB | DT | RF |
|---|---|---|---|---|---|
| **Nguyên lý** | Linear boundary | Distance-based | Probabilistic | Rule-based | Ensemble |
| **Tốc độ train** | Nhanh | Gần 0 | Rất nhanh | Nhanh | Chậm (200 cây) |
| **Tốc độ predict** | Nhanh | Chậm (tìm hàng xóm) | Nhanh | Nhanh | Trung bình |
| **Interpretability** | Cao | Trung bình | Cao | Cao | Thấp (black box) |
| **Imbalance handling** | `class_weight` | `distance weight` | Prior estimate | `class_weight` | `class_weight` |
| **Kết quả AUC** | 0.9112 | 0.8521 | 0.8771 | 0.7787 | **0.9264** |

**Kết quả đầy đủ 5 model:**

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 80.46% | 86.72% | 77.98% | 0.9112 |
| KNN | 81.01% | 66.81% | 86.62% | 0.8521 |
| Naive Bayes | 80.10% | 78.59% | 80.69% | 0.8771 |
| Decision Tree | 82.10% | 68.09% | 87.64% | 0.7787 |
| **Random Forest** | **86.65%** | **74.52%** | **91.45%** | **0.9264** |

Random Forest thắng về Accuracy, Specificity và AUC. Logistic Regression có Sensitivity cao nhất (86.72%) — phù hợp cho bài toán cần detect nhiều Churn nhất.
