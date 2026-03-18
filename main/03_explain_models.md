# Giải thích: `models.py` — Bước 3: Định Nghĩa & Huấn Luyện Mô Hình

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py  [✓]
      └─► data_preprocessing.py  [✓]
           └─► [BƯỚC 3] models.py  ← Bạn đang ở đây
                └─► evaluation.py
                     └─► explainability.py
```

---

## Tổng Quan

File này định nghĩa và huấn luyện **5 thuật toán Machine Learning** đúng như bài báo liệt kê trong Section 3.3. Ngoài ra, code thêm bước **GridSearchCV** để tự động tìm siêu tham số tốt nhất cho Random Forest — điểm này vượt ra ngoài bài báo để cải thiện kết quả.

**Đầu vào:** `X_train` (numpy array), `y_train` (numpy array)  
**Đầu ra:** Dictionary `{tên_model: model_đã_train}`

---

## Liên Hệ Bài Báo (Section 3.3)

Bài báo (Section 3.3 — "Machine Learning Algorithms") liệt kê chính xác 5 thuật toán:

| Bài báo mô tả | Class trong scikit-learn | Trong code |
|---|---|---|
| Logistic Regression | `LogisticRegression` | ✓ |
| K-Nearest Neighbor (KNN) Classifier | `KNeighborsClassifier` | ✓ |
| Naïve Bayes Classifier | `GaussianNB` | ✓ |
| Decision Tree Classifier | `DecisionTreeClassifier` | ✓ |
| Random Forest Classifier | `RandomForestClassifier` | ✓ |

---

## Giải Thích Từng Thuật Toán

### 1. Logistic Regression

```python
LogisticRegression(
    max_iter=1000,
    C=0.5,          # Hệ số regularization (nhỏ = regularization mạnh hơn)
    solver='lbfgs',
    random_state=42
)
```

**Bài báo mô tả:**
> *"Logistical models attempt to create a regression model based on data with the binary response variable. In Logistic Regression, the logit function is used to determine the probability of a binary outcome."*

**Nguyên lý:** Mô hình tính xác suất khách hàng churn bằng hàm sigmoid:

$$P(\text{Churn}=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + ... + \beta_n x_n)}}$$

- Nếu $P > 0.5$ → dự đoán **Churn (1)**
- Nếu $P \leq 0.5$ → dự đoán **Not Churn (0)**

**Hyperparameter trong code:**
- `C=0.5`: Regularization L2, ngăn overfit
- `max_iter=1000`: Đủ bước để hội tụ với dataset lớn
- `solver='lbfgs'`: Thuật toán tối ưu phù hợp với dataset nhỏ/vừa

---

### 2. KNN (K-Nearest Neighbor)

```python
KNeighborsClassifier(
    n_neighbors=7,        # Số láng giềng gần nhất
    weights='distance',   # Trọng số ngược chiều khoảng cách
    metric='minkowski'    # Khoảng cách Minkowski (= Euclidean khi p=2)
)
```

**Bài báo mô tả:**
> *"KNN is a non-parametric algorithm... The KNN algorithm assumes that the new data is similar to the existing data and assigns the new data to the category that is most similar to the existing category based on the distance between two points."*

**Nguyên lý:** Với mỗi điểm test, tìm 7 điểm train gần nhất, vote theo khoảng cách:

```
Điểm test mới → Tính khoảng cách tới TẤT CẢ điểm train
               → Lấy 7 điểm gần nhất
               → Điểm gần hơn được trọng số lớn hơn (weights='distance')
               → Class nào có tổng trọng số lớn hơn → Dự đoán class đó
```

**Lý do chọn `n_neighbors=7` (thay vì 5 mặc định):**
- Số lẻ để tránh hòa (tie)
- n=7 giúp mô hình ổn định hơn, ít bị ảnh hưởng bởi nhiễu

---

### 3. Naïve Bayes (Gaussian)

```python
GaussianNB(
    var_smoothing=1e-8   # Làm mịn để tránh xác suất = 0
)
```

**Bài báo mô tả:**
> *"This procedure is based on Bayes' Theorem and the assumption that the predictors are unrelated... particularly successful when dealing with large datasets."*

**Nguyên lý — Định lý Bayes:**

$$P(\text{Churn} | \mathbf{x}) = \frac{P(\mathbf{x} | \text{Churn}) \cdot P(\text{Churn})}{P(\mathbf{x})}$$

Giả định "Naïve" (ngây thơ): Các feature **độc lập với nhau**, vì vậy:

$$P(\mathbf{x} | \text{Churn}) = \prod_{i=1}^{n} P(x_i | \text{Churn})$$

**GaussianNB** giả định mỗi feature tuân theo phân phối chuẩn (Gaussian) trong mỗi class.

**Ưu điểm:** Nhanh, ít dữ liệu, hoạt động tốt với các feature thực sự độc lập  
**Nhược điểm:** Giả định độc lập thường không đúng trong thực tế

---

### 4. Decision Tree

```python
DecisionTreeClassifier(
    max_depth=15,          # Độ sâu tối đa của cây
    min_samples_split=5,   # Cần ít nhất 5 mẫu để tách nút
    min_samples_leaf=2,    # Mỗi lá cần ít nhất 2 mẫu
    criterion='entropy',   # Tiêu chí tách: Information Gain
    random_state=42
)
```

**Bài báo mô tả:**
> *"These trees comprise a root at the top and knots that are interconnected by branches. At each internal node, a specific attribute is tested, and the result guides the selection of different branches, eventually leading to a terminal node."*

**Nguyên lý:** Tại mỗi nút, chọn feature và ngưỡng tách để tối đa hóa **Information Gain**:

$$\text{IG}(S, A) = H(S) - \sum_{v \in A} \frac{|S_v|}{|S|} H(S_v)$$

Trong đó entropy $H(S) = -\sum_i p_i \log_2 p_i$

**Ví dụ cây quyết định:**
```
Contract == 'Month-to-Month'?
├── Yes → Tenure < 12 months?
│         ├── Yes → CHURN (xác suất cao)
│         └── No  → Monthly Charge > 90$?
│                   ├── Yes → CHURN
│                   └── No  → NOT CHURN
└── No → NOT CHURN (hợp đồng dài hạn ít churn)
```

**Tham số kiểm soát overfitting:**
- `max_depth=15`: Cây không quá sâu → tránh học thuộc training data
- `min_samples_split=5`, `min_samples_leaf=2`: Ràng buộc tại các nút lá

---

### 5. Random Forest ⭐ (Mô hình tốt nhất)

```python
RandomForestClassifier(
    n_estimators=200,    # Số cây trong rừng
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt', # Mỗi cây chỉ dùng √25 ≈ 5 features ngẫu nhiên
    bootstrap=True,      # Lấy mẫu có hoàn lại (bagging)
    random_state=42,
    n_jobs=-1            # Dùng tất cả CPU cores
)
```

**Bài báo mô tả:**
> *"A Random Forest is a combination method that works with Decision Trees as building blocks. The algorithm generates a predefined number of trees and takes a cut of the total number of trees and uses it as its predictor."*

**Nguyên lý — Ensemble Learning (Bagging):**

```
Training data (5282 mẫu)
        │
        ├─ Bootstrap sample 1 → Decision Tree 1 → Dự đoán 1
        ├─ Bootstrap sample 2 → Decision Tree 2 → Dự đoán 2
        ├─ ...
        └─ Bootstrap sample 200 → Decision Tree 200 → Dự đoán 200
                                                              │
                                                    Majority Vote
                                                              │
                                                    Kết quả cuối
```

**Hai nguồn ngẫu nhiên hóa:**
1. **Bootstrap**: Mỗi cây train trên ~63% mẫu (lấy ngẫu nhiên, có hoàn lại)
2. **Feature sampling**: Mỗi lần tách nút chỉ xét $\sqrt{25} \approx 5$ features ngẫu nhiên

→ 200 cây **đa dạng**, khi vote đa số sẽ **chính xác hơn** bất kỳ cây đơn nào.

---

### GridSearchCV — Tự Động Tìm Siêu Tham Số

```python
def get_tuned_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        rf, param_grid,
        cv=5,              # 5-fold cross validation
        scoring='roc_auc', # Tối ưu theo AUC
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
```

**Kết quả thực tế sau khi chạy GridSearchCV:**
```
Tuning Random Forest with GridSearchCV...
  Best params: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
  Best CV AUC: 0.9621
```

→ GridSearchCV tìm ra `max_depth=20` (sâu hơn mặc định 15), `min_samples_leaf=1`, `min_samples_split=2` — cho phép từđến tận cành lá đơn lẻ. AUC cross-validation = **0.9621**.

---

### Tầm quan trọng của con số 0.9621

CV AUC = 0.9621 là trước SMOTE training hoàn chỉnh. AUC thực trên test set sau cùng là **0.8892** (vì test set không qua SMOTE, phản ánh phân phối thực tế).

**5-Fold Cross Validation:**
```
Train set (5282 mẫu) chia thành 5 fold:
│ Fold 1 │ Fold 2 │ Fold 3 │ Fold 4 │ Fold 5 │

Vòng 1: Train [F2,F3,F4,F5], Validate [F1]
Vòng 2: Train [F1,F3,F4,F5], Validate [F2]
...
Vòng 5: Train [F1,F2,F3,F4], Validate [F5]
                                    │
                            AUC trung bình 5 vòng
```

→ Tìm tổ hợp hyperparameter có **AUC trung bình cao nhất** → Dùng để train mô hình cuối.

---

## Hàm `train_all_models()` — Điều Phối

```python
def train_all_models(X_train, y_train, tune_rf=True):
    models = get_models()
    trained_models = {}
    
    for name, model in models.items():
        if name == 'Random Forest' and tune_rf:
            # Dùng GridSearchCV để tune trước
            trained_model = get_tuned_random_forest(X_train, y_train)
        else:
            # Train thẳng với hyperparameter đã định sẵn
            model.fit(X_train, y_train)
            trained_model = model
        
        trained_models[name] = trained_model
    
    return trained_models
```

---

## Hàm `predict_all_models()` — Lấy Dự Đoán

```python
def predict(model, X):
    y_pred = model.predict(X)           # Class: 0 hoặc 1
    y_prob = model.predict_proba(X)[:,1]  # Xác suất của class 1 (Churn)
    return y_pred, y_prob
```

**Tại sao cần cả `y_pred` và `y_prob`?**

| | Dùng cho |
|---|---|
| `y_pred` (0/1) | Confusion matrix, Accuracy, Sensitivity, Specificity |
| `y_prob` (0.0–1.0) | ROC curve, AUC — cần xác suất liên tục để vẽ đường cong |

---

## Bảng So Sánh 5 Thuật Toán

| Thuật toán | Loại | Ưu điểm | Nhược điểm | Kết quả (thực tế) |
|---|---|---|---|---|
| Logistic Regression | Linear | Đơn giản, diễn giải được | Chỉ tốt với quan hệ tuyến tính | Acc: 77.00%, AUC: 0.8837 |
| KNN | Instance-based | Không giả định gì, linh hoạt | Chậm lúc predict, nhạy cảm với scale | Acc: 72.63%, AUC: 0.8271 |
| Naive Bayes | Probabilistic | Cực nhanh, ít dữ liệu | Giả định độc lập sai | Acc: 76.21%, AUC: 0.8478 |
| Decision Tree | Tree | Dễ diễn giải | Dễ overfit | Acc: 79.78%, AUC: 0.7918 |
| **Random Forest** | **Ensemble** | **Chính xác, robust** | Khó diễn giải (→ cần LIME/SHAP) | **Acc: 82.28%, AUC: 0.8892** |

> **So sánh với bài báo (không dùng SMOTE):** Bài báo ra
> RF Acc=86.94%, AUC=0.95. Code này dùng SMOTE cân bằng dữ liệu nên Sensitivity tăng nhưng Accuracy tổng thể thấp hơn do test set vẫn là phân phối thực (imbalanced).

---

*Trước: [02_explain_data_preprocessing.md](02_explain_data_preprocessing.md)*  
*Tiếp theo: [04_explain_evaluation.md](04_explain_evaluation.md)*
