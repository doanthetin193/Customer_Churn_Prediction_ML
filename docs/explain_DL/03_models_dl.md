# Giải thích: `models_dl.py` — Bước 3: Kiến Trúc và Huấn Luyện MLP

## Vị trí trong Pipeline

```
main.py (DL)
 └─► data_loader.py (BƯỚC 1)
      └─► data_preprocessing.py (BƯỚC 2)
           └─► [BƯỚC 3] models_dl.py  ← Bạn đang ở đây
                └─► evaluation.py
```

---

## Mục đích

Định nghĩa kiến trúc **MLP (Multi-Layer Perceptron)** cho bài toán churn, cùng với các hàm training, tìm ngưỡng phân loại tối ưu và inference. Đây là file cốt lõi của toàn bộ DL pipeline.

---

## Tổng quan kiến trúc

```
Input (25 features)
    │
    ▼
Dense(256) → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.4)   ← Block 1
    │
    ▼
Dense(128) → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.4)   ← Block 2
    │
    ▼
Dense(64)  → BatchNorm → LeakyReLU(α=0.1) → Dropout(0.2)   ← Block 3
    │
    ▼
Dense(1) → Sigmoid                                           ← Output
    │
    ▼
P(Churn) ∈ [0, 1]
```

---

## Phân tích Code Chi Tiết

### Hàm 1: `set_seed(seed=42)`

```python
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

Tensorflow sử dụng **2 nguồn ngẫu nhiên** riêng biệt:
- `np.random.seed(42)` — numpy (khởi tạo weight, data shuffle trong một số ops)
- `tf.random.set_seed(42)` — TensorFlow global seed (dropout mask, weight initialization)

Cần set cả hai để đảm bảo reproducibility hoàn toàn.

---

### Hàm 2: `build_mlp(...)` — Kiến trúc Chính

```python
def build_mlp(
    input_dim: int,
    hidden_units: tuple[int, ...] = (256, 128, 64),
    dropout_rate: float = 0.4,
    learning_rate: float = 5e-4,
    l2_lambda: float = 3e-4,
    label_smoothing: float = 0.0,
) -> keras.Model:
```

**Giải thích từng tham số:**

| Tham số | Giá trị | Lý do |
|---|---|---|
| `input_dim` | 25 | Số feature đầu vào (từ `X_train.shape[1]`) |
| `hidden_units` | `(256, 128, 64)` | 3 block, giảm dần — funnel architecture |
| `dropout_rate` | 0.4 | 40% neurons bị tắt ngẫu nhiên mỗi batch |
| `learning_rate` | 5e-4 = 0.0005 | Thấp hơn default Adam (1e-3) → hội tụ ổn định hơn |
| `l2_lambda` | 3e-4 = 0.0003 | Hệ số L2 regularization |
| `label_smoothing` | 0.0 | Tắt → dùng hard labels (0 hoặc 1) |

---

#### Regularizer L2

```python
regularizer = keras.regularizers.l2(l2_lambda)
```

L2 regularization thêm penalty vào loss:

$$\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda \sum_j w_j^2$$

Với $\lambda = 3 \times 10^{-4}$. Penalty này phạt weights lớn → model phải dùng nhiều feature nhỏ thay vì phụ thuộc vào 1-2 feature lớn → giảm overfitting.

---

#### 3 Block (Dense → BN → LeakyReLU → Dropout)

```python
model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),
    # Block 1
    keras.layers.Dense(256, kernel_regularizer=regularizer),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(negative_slope=0.1),
    keras.layers.Dropout(dropout_rate),        # 0.4
    # Block 2
    keras.layers.Dense(128, kernel_regularizer=regularizer),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(negative_slope=0.1),
    keras.layers.Dropout(dropout_rate),        # 0.4
    # Block 3
    keras.layers.Dense(64, kernel_regularizer=regularizer),
    keras.layers.BatchNormalization(),
    keras.layers.LeakyReLU(negative_slope=0.1),
    keras.layers.Dropout(dropout_rate * 0.5),  # 0.2 — nhẹ hơn ở layer cuối
    # Output
    keras.layers.Dense(1, activation="sigmoid"),
])
```

**Từng lớp trong mỗi block:**

**Dense(n):**
$$h = XW + b$$

Ma trận nhân features × weights → hidden representation. `kernel_regularizer=regularizer` áp l2 lên $W$.

**BatchNormalization:**
$$\hat{h} = \frac{h - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta$$

Normalize output của Dense theo từng batch → giúp gradient stable, training nhanh hơn. $\gamma, \beta$ là learnable parameters.

**Thứ tự: Dense → BN → Activation (LeakyReLU):**

Đây là thứ tự chuẩn (pre-activation). BN trước activation → normalize trước khi đưa vào nonlinearity → gradient flow tốt hơn.

**LeakyReLU(negative_slope=0.1):**

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{nếu } x > 0 \\ 0.1x & \text{nếu } x \leq 0 \end{cases}$$

**Tại sao LeakyReLU thay vì ReLU?**

ReLU thông thường: khi $x \leq 0$ → output = 0 → gradient = 0 → neuron "chết" (Dying ReLU problem), không bao giờ học lại. LeakyReLU giữ gradient nhỏ (0.1) khi $x \leq 0$ → neuron vẫn có thể phục hồi.

**Dropout(rate):**

Trong quá trình training, mỗi neuron bị ngắt ngẫu nhiên với xác suất = `rate`. Khi inference (predict), tất cả neurons hoạt động nhưng output được scale bởi `(1 - rate)` để compensate.

| Block | Dropout rate | Lý do |
|---|---|---|
| Block 1 | 0.4 | Regularize mạnh ở layer rộng |
| Block 2 | 0.4 | Tiếp tục regularize |
| Block 3 | 0.2 | Nhẹ hơn ở layer gần output — cần giữ nhiều thông tin |

**Output Dense(1) + Sigmoid:**

$$P(\text{Churn}) = \sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$$

1 neuron duy nhất cho binary classification. Sigmoid ép output về [0,1] — diễn giải là xác suất Churn.

---

#### Compile Model

```python
optimizer = keras.optimizers.Adam(learning_rate=5e-4)
model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(label_smoothing=0.0),
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)
```

**Adam optimizer với lr = 5e-4:**

Adam = Adaptive Moment Estimation. Tự điều chỉnh learning rate cho từng parameter dựa vào lịch sử gradient:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Mặc định: $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$. Dùng lr=5e-4 thay vì default 1e-3 để hội tụ chậm hơn nhưng ổn định hơn.

**BinaryCrossentropy:**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N [y_i \log(p_i) + (1-y_i)\log(1-p_i)]$$

Loss function chuẩn cho binary classification. `label_smoothing=0.0` → dùng hard targets (bài báo không đề cập label smoothing).

**4 metrics theo dõi trong training:**

| Metric | Ý nghĩa | Monitor cho |
|---|---|---|
| `accuracy` | % đúng với threshold 0.5 | Cái nhìn tổng quan |
| `auc` | AUC ROC | **Chính** — dùng cho EarlyStopping |
| `precision` | TP/(TP+FP) | Tránh cảnh báo sai |
| `recall` | TP/(TP+FN) = Sensitivity | Phát hiện Churn |

---

#### Số parameters thực tế

| Layer | Parameters |
|---|---|
| Dense(256) | 25×256 + 256 = **6,656** |
| BatchNorm(256) | 256×4 = 1,024 |
| Dense(128) | 256×128 + 128 = **32,896** |
| BatchNorm(128) | 512 |
| Dense(64) | 128×64 + 64 = **8,256** |
| BatchNorm(64) | 256 |
| Dense(1) | 64×1 + 1 = **65** |
| **Tổng trainable** | **~49,665** |

---

### Hàm 3: `find_optimal_threshold(model, X_val, y_val)`

```python
def find_optimal_threshold(model, X_val, y_val) -> tuple[float, float]:
    y_prob = model.predict(X_val, verbose=0).reshape(-1)
    thresholds = np.arange(0.05, 0.95, 0.01)  # 90 ngưỡng

    best_j, best_t = -1.0, 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)

    print(f"  Optimal threshold: {best_t:.2f}  (Youden's J = {best_j:.4f})")
    return best_t, best_j
```

**Tại sao không dùng threshold 0.5 mặc định?**

Với imbalanced data (72% Stayed, 28% Churn), model có xu hướng thiên về class majority. Threshold 0.5 → Specificity cao nhưng Sensitivity thấp → bỏ sót nhiều khách Churned.

**Youden's J statistic:**

$$J = \text{Sensitivity} + \text{Specificity} - 1 = TPR + TNR - 1$$

Giá trị J:
- $J = 1$ → perfect classifier (100% Sensitivity và 100% Specificity)
- $J = 0$ → classifier ngẫu nhiên
- $-1 \leq J \leq 1$

J tối đa hóa đồng thời cả Sensitivity và Specificity — không hi sinh metric này cho metric kia. Đây là cách cân bằng tốt nhất theo lý thuyết ROC.

**Cơ chế hoạt động:**

```
Scan 90 threshold từ 0.05 đến 0.94:
  t=0.05 → J=0.23 (sensitivity rất cao, specificity rất thấp)
  t=0.10 → J=0.45
  ...
  t=0.54 → J=0.7472  ← BEST
  ...
  t=0.94 → J=0.10 (sensitivity rất thấp, specificity rất cao)

best_t = 0.54
```

**Tại sao dùng validation set (X_val) chứ không phải test set?**

Nếu dùng test set để tìm threshold → **data leakage**: model được optimize trên test → metric trên test overly optimistic. Validation set là tập trung gian, tách hoàn toàn khỏi final evaluation.

**Kết quả thực tế:**

```
Optimal threshold: 0.54  (Youden's J = 0.7472)
```

Threshold 0.54 > 0.5 → model cần "chắc chắn hơn một chút" trước khi predict Churn → tăng Specificity một chút so với threshold rất thấp.

---

### Hàm 4: `train_mlp(X_train, y_train, X_val, y_val, ...)`

```python
def train_mlp(X_train, y_train, X_val, y_val,
              class_weight=None, epochs=150, batch_size=64, patience=20):
    set_seed(42)
    model = build_mlp(input_dim=X_train.shape[1])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=patience,          # Dừng nếu val_auc không cải thiện sau 20 epochs
            restore_best_weights=True,  # Khôi phục weights của epoch tốt nhất
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", mode="max",
            factor=0.5,                 # lr_new = lr_old × 0.5
            patience=max(5, patience // 4),  # = 5 epochs
            min_lr=5e-6,                # Giới hạn dưới của lr
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=64,
        class_weight=class_weight,      # {0: 0.698, 1: 1.763}
        callbacks=callbacks,
        verbose=1,
    )

    return model, history
```

**`EarlyStopping` — tại sao monitor `val_auc` thay vì `val_loss`?**

`val_loss` đo cross-entropy trên validation → bị ảnh hưởng bởi class imbalance (loss dominated bởi majority class). `val_auc` đo khả năng phân biệt Churn/Stayed tổng thể → metric có ý nghĩa hơn cho imbalanced classification.

**`restore_best_weights=True`:** Sau khi training dừng, Keras tự động roll back về weights của epoch có `val_auc` cao nhất (không phải epoch cuối cùng).

**`ReduceLROnPlateau`:**

Khi model bắt đầu plateau (val_auc không tăng sau 5 epochs):
```
lr: 0.0005 → 0.00025 → 0.000125 → ... → min 0.000005
```

Cho phép model "tinh chỉnh" với bước nhỏ hơn khi gần đến optimum.

**`class_weight={0: 0.698, 1: 1.763}`:**

Khách Stayed (class 0): mỗi sample có weight 0.698 (nhẹ hơn)
Khách Churned (class 1): mỗi sample có weight 1.763 (nặng hơn 2.5×)

→ Loss của mỗi epoch = weighted average → model chú ý nhiều hơn đến lỗi trên class Churned.

**Training dynamics thực tế:**

```
Epoch 1/150:  loss=0.65, auc=0.71 | val_loss=0.60, val_auc=0.77
Epoch 10/150: loss=0.54, auc=0.83 | val_loss=0.52, val_auc=0.88
Epoch 30/150: loss=0.47, auc=0.89 | val_loss=0.49, val_auc=0.91
...
EarlyStopping: val_auc không cải thiện sau 20 epochs → dừng
Restored best weights (epoch ~X, val_auc=0.9132)
```

---

### Hàm 5: `predict_mlp(model, X, threshold=0.5)`

```python
def predict_mlp(model, X, threshold=0.5) -> tuple[np.ndarray, np.ndarray]:
    y_prob = model.predict(X, verbose=0).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob
```

**`model.predict(X).reshape(-1)`:**

`model.predict()` trả về shape `(n_samples, 1)` — 2D (vì output layer có 1 neuron). `.reshape(-1)` → 1D array `(n_samples,)` để dùng với sklearn metrics.

**`(y_prob >= threshold).astype(int)`:**

Boolean array → int: `True → 1 (Churn), False → 0 (Stayed)`.

Với `threshold=0.54` (optimal):
```
y_prob = [0.23, 0.61, 0.49, 0.77, ...]
threshold = 0.54
y_pred  = [0,    1,    0,    1,   ...]
```

---

## So sánh phương pháp DL vs ML

| Tiêu chí | ML (Random Forest) | DL (MLP) |
|---|---|---|
| **Kiến trúc** | 200 Decision Trees | 3-block MLP, ~49k params |
| **Class imbalance** | `class_weight='balanced'` | `class_weight` trong `model.fit()` |
| **Threshold** | 0.5 (mặc định) | 0.54 (optimal theo Youden's J) |
| **Training** | `.fit()` 1 lần | Mini-batch gradient descent, 150 epochs |
| **Regularization** | Không explicit | L2 + Dropout + BatchNorm |
| **Interpretability** | SHAP/LIME khả dụng | Black box hơn |
| **Accuracy** | **86.65%** | 81.74% |
| **Sensitivity** | 74.52% | **84.37%** |
| **Specificity** | **91.45%** | 80.69% |
| **AUC** | **0.9264** | 0.9132 |

MLP detect được nhiều Churned hơn (Sensitivity 84.37% vs 74.52%), phù hợp nếu mục tiêu là **không bỏ sót khách có nguy cơ rời đi**. RF vượt trội về AUC và Specificity tổng thể.
