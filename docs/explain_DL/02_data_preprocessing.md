# Giải thích: `data_preprocessing.py` — Bước 2: Tiền Xử Lý Dữ Liệu (DL Pipeline)

## Vị trí trong Pipeline

```
main.py (DL)
 └─► data_loader.py (BƯỚC 1)
      └─► [BƯỚC 2] data_preprocessing.py  ← Bạn đang ở đây
           └─► models_dl.py
                └─► evaluation.py
```

---

## Mục đích

Biến đổi DataFrame thô (7043×38) thành các tensor numpy sẵn sàng đưa vào MLP. Pipeline này **giống hệt** ML pipeline về logic nhưng có một số cải tiến nhỏ về code style và Python type hints.

---

## So sánh với ML Pipeline

| Tiêu chí | ML (`main/data_preprocessing.py`) | DL (`main_DL/data_preprocessing.py`) |
|---|---|---|
| Logic pipeline | 7 hàm, 8 bước | 7 hàm, 7 bước (giống nhau) |
| Filter "Joined" | ✅ Có | ✅ Có (cùng fix) |
| use_smote | False | False |
| `fillna(inplace=True)` | ✅ → FutureWarning | `data[col] = data[col].fillna(...)` → không có warning |
| Type hints | Cơ bản | Đầy đủ với `tuple[pd.DataFrame, dict[str, LabelEncoder]]` |
| `from __future__ import annotations` | Không có | Có — cho phép dùng `X \| Y` union type trên Python 3.9 |
| Key dict trả về | snake_case | camelCase-ish, `"X_train"` | Giống nhau |

---

## Liên hệ với Bài Báo

| Mô tả trong bài báo | Tương ứng trong code |
|---|---|
| Loại "Joined" — khách mới chưa có hành vi churn | `create_target_variable()` filter |
| 25 features được chọn | `select_features()` |
| Split 75/25 | `train_test_split(test_size=0.25)` |
| StandardScaler | `scale_features()` |
| Không dùng SMOTE | `use_smote=False` |

---

## Phân tích Code Chi Tiết

### Hàm 1: `create_target_variable(df)`

```python
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data[data["Customer Status"].isin(["Churned", "Stayed"])].reset_index(drop=True)
    data["Churn"] = (data["Customer Status"] == "Churned").astype(int)
    print(f"Filtered to Churned+Stayed only: {len(data)} customers")
    return data
```

Lọc "Joined" → 7043 hàng thành **6589 hàng**. Tạo cột `Churn` nhị phân (0/1).

| Customer Status | Số lượng | Nhãn |
|---|---|---|
| Stayed | 4720 | Churn = 0 |
| Churned | 1869 | Churn = 1 |
| Joined | 454 | ❌ Bị loại |

**Tỉ lệ churn sau lọc: 1869 / 6589 = 28.4%**

---

### Hàm 2: `select_features(df)`

```python
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "Gender", "Age", "Married", "Number of Dependents",
        "Number of Referrals", "Tenure in Months", "Offer",
        "Phone Service", "Multiple Lines", "Internet Service", "Internet Type",
        "Online Security", "Online Backup", "Device Protection Plan",
        "Premium Tech Support", "Streaming TV", "Streaming Movies",
        "Streaming Music", "Unlimited Data", "Contract", "Paperless Billing",
        "Payment Method", "Monthly Charge", "Total Charges", "Total Revenue"
    ]
    available_cols = [col for col in feature_cols if col in df.columns]
    return df[available_cols + ["Churn"]]
```

**25 features giống hệt với ML pipeline** — quan trọng để so sánh công bằng giữa DL và ML. Nếu DL dùng nhiều feature hơn thì performance tốt hơn chưa chắc là vì kiến trúc tốt hơn.

---

### Hàm 3: `handle_missing_values(df)`

```python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    categorical_cols = data.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].median())  # ← Không dùng inplace

    for col in categorical_cols:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mode().iloc[0])  # ← iloc[0] thay vì [0]
    return data
```

**Cải tiến so với ML pipeline:**

| Vấn đề | ML pipeline | DL pipeline |
|---|---|---|
| `fillna(inplace=True)` | Có → FutureWarning trong pandas 3.x | Không dùng → `data[col] = data[col].fillna(...)` |
| `mode()[0]` | Có thể warning nếu Series rỗng | `mode().iloc[0]` — dùng positional index, tường minh hơn |

Cả hai cách đều cho kết quả giống nhau, nhưng DL pipeline dùng cú pháp pandas hiện đại hơn.

---

### Hàm 4: `encode_categorical(df)`

```python
def encode_categorical(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    data = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[col] = encoder.fit_transform(data[col].astype(str))
        encoders[col] = encoder
    return data, encoders
```

`tuple[pd.DataFrame, dict[str, LabelEncoder]]` — type hint tường minh cho return type. Nhờ `from __future__ import annotations` ở đầu file, syntax này hoạt động trên Python 3.9+ (không cần 3.10).

Logic giống ML pipeline: LabelEncoder → object columns → int.

---

### Hàm 5: `scale_features(X_train, X_test)`

```python
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
```

**Cực kỳ quan trọng với Neural Network:**

StandardScaler đặc biệt cần thiết cho MLP hơn so với ML models:

| Lý do | Giải thích |
|---|---|
| Gradient descent | Optimizer Adam update weights theo gradient. Feature với scale lớn (Total Revenue có thể > 10000) → gradient lớn → learning rate hiệu quả khác nhau theo từng feature |
| Khởi tạo weight | Default Glorot initialization giả định input ~ N(0,1) |
| Batch Normalization | Hoạt động tốt hơn khi input đã normalized sẵn |

Feature không được scale → training không ổn định, loss dao động mạnh, model học chậm hoặc không hội tụ.

---

### Hàm 6: `preprocess_data(df, ...)` — Hàm Chính

```python
def preprocess_data(df, test_size=0.25, random_state=42, use_smote=False) -> dict:
    # Bước 1-4: transform
    data = create_target_variable(df)
    data = select_features(data)
    data = handle_missing_values(data)
    data, encoders = encode_categorical(data)

    # Bước 5: tách X, y
    X = data.drop(columns=["Churn"])
    y = data["Churn"]
    feature_names = list(X.columns)

    # Bước 6: Split 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Bước 7: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # SMOTE: không dùng
    if use_smote: ...

    # Chuyển sang numpy
    y_train_array = y_train if isinstance(y_train, np.ndarray) else y_train.to_numpy()

    return {
        "X_train": X_train_scaled, "X_test": X_test_scaled,
        "y_train": y_train_array,  "y_test": y_test.to_numpy(),
        "feature_names": feature_names,
        ...
    }
```

**`y_train.to_numpy()` thay vì `.values`:**

Pandas khuyến khích dùng `.to_numpy()` từ version 0.24 thay cho `.values` — tường minh hơn về dtype.

**Lưu ý quan trọng: `preprocess_data()` chỉ trả về split train/test 75/25.**

Trong DL pipeline, `main.py` sẽ **tách thêm** một lần nữa để tạo validation set từ tập train:

```python
# Trong main.py (DL)
data = preprocess_data(df)          # Train=4941, Test=1648
X_train, X_val, y_train, y_val = train_test_split(
    data["X_train"], data["y_train"],
    test_size=0.2, stratify=data["y_train"]  # 20% của train = val
)
# → Train (final)=3953, Val=988, Test=1648
```

**ML pipeline không cần validation set** (sklearn model không cần) nhưng DL cần để theo dõi `val_auc` trong quá trình training và trigger EarlyStopping.

---

## Luồng dữ liệu tóm tắt

```
DataFrame(7043×38)
    ↓ create_target_variable()  → 6589 hàng, thêm cột 'Churn'
    ↓ select_features()         → 25 feature đã chọn
    ↓ handle_missing_values()   → 0 missing values
    ↓ encode_categorical()      → tất cả columns là số
    ↓ train_test_split(75/25)   → Train=4941, Test=1648
    ↓ scale_features()          → μ=0, σ=1

Output từ preprocess_data():
  X_train: (4941, 25) — sẽ được split tiếp trong main.py
  X_test:  (1648, 25) — giữ nguyên làm final test
  y_train: (4941,)
  y_test:  (1648,)

Sau khi main.py split thêm (val 20%):
  X_train (final): (3953, 25) — dùng để fit model
  X_val:           (988, 25)  — dùng để monitor training & find threshold
  X_test:          (1648, 25) — dùng để báo cáo kết quả cuối
```

**Tỉ lệ churn duy trì qua các split (stratify=True):**

| Tập | Kích thước | Churn rate |
|---|---|---|
| Toàn bộ (sau filter) | 6589 | 28.4% |
| Train (75%) | 4941 | 28.4% |
| Train final (80% của train) | 3953 | 28.4% |
| Validation (20% của train) | 988 | 28.4% |
| Test (25%) | 1648 | 28.4% |
