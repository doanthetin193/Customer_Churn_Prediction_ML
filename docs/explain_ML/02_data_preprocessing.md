# Giải thích: `data_preprocessing.py` — Bước 2: Tiền Xử Lý Dữ Liệu

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py (BƯỚC 1)
      └─► [BƯỚC 2] data_preprocessing.py  ← Bạn đang ở đây
           └─► models.py
                └─► evaluation.py
                     └─► explainability.py
```

---

## Mục đích

Biến đổi DataFrame thô (7043 hàng, 38 cột) thành ma trận số đã chuẩn hoá sẵn sàng đưa vào model. Thực hiện **8 bước** theo thứ tự nghiêm ngặt.

---

## Liên hệ với Bài Báo (Section 3.2 — Data Preprocessing)

| Mô tả trong bài báo | Tương ứng trong code |
|---|---|
| *"The dataset consists of 7043 rows and 38 attributes"* | Input: `df.shape = (7043, 38)` |
| *"Joined" customers excluded* (không churn, không stay) | `create_target_variable()` — filter chỉ giữ Churned + Stayed |
| *"25 features were selected"* | `select_features()` — trả về 25 cột |
| *"split 75-25 for train/test"* | `train_test_split(test_size=0.25)` |
| *"StandardScaler applied"* | `scale_features()` → `StandardScaler` |
| *"Label encoding used for categorical"* | `encode_categorical()` → `LabelEncoder` |
| Bài báo KHÔNG dùng SMOTE | `use_smote=False` (mặc định) |

---

## Phân tích Code Chi Tiết

### Hàm 1: `create_target_variable(df)`

```python
def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df['Customer Status'].isin(['Churned', 'Stayed'])].reset_index(drop=True)
    df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)
    print(f"✓ Filtered to Churned+Stayed only: {len(df)} customers")
    return df
```

**Tại sao phải lọc "Joined"?**

Khách hàng có `Customer Status = 'Joined'` là **khách hàng vừa gia nhập**. Họ không có lịch sử sử dụng đủ để predict churn, và nhãn của họ không phải 0 (Stayed) cũng chẳng phải 1 (Churned) — nên nếu giữ lại sẽ tạo noise cho model.

| Customer Status | Số lượng | Giữ lại? |
|---|---|---|
| Stayed | 4720 | ✅ → Churn = 0 |
| Churned | 1869 | ✅ → Churn = 1 |
| Joined | 454 | ❌ Bị loại |

Kết quả: 7043 → **6589 khách hàng**, tỉ lệ churn = **28.4%**

**Từng dòng:**

| Dòng | Ý nghĩa |
|---|---|
| `df.copy()` | Tránh sửa DataFrame gốc (immutable pattern) |
| `df['Customer Status'].isin([...])` | Boolean mask: True nếu là Churned hoặc Stayed |
| `.reset_index(drop=True)` | Đánh lại index từ 0 sau khi xoá hàng |
| `(df['...'] == 'Churned').astype(int)` | Boolean → int: True=1, False=0 |

---

### Hàm 2: `select_features(df)`

```python
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        'Gender', 'Age', 'Married', 'Number of Dependents',
        'Number of Referrals', 'Tenure in Months', 'Offer',
        'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
        'Online Security', 'Online Backup', 'Device Protection Plan',
        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
        'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
        'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Revenue'
    ]
    available_cols = [c for c in feature_cols if c in df.columns]
    return df[available_cols + ['Churn']]
```

**25 features được chọn — phân loại:**

| Nhóm | Features |
|---|---|
| **Nhân khẩu học** | Gender, Age, Married, Number of Dependents |
| **Tài khoản** | Number of Referrals, Tenure in Months, Offer, Contract, Paperless Billing, Payment Method |
| **Dịch vụ điện thoại** | Phone Service, Multiple Lines |
| **Dịch vụ Internet** | Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| **Tài chính** | Monthly Charge, Total Charges, Total Revenue |

`available_cols = [c for c in feature_cols if c in df.columns]` — list comprehension phòng trường hợp dataset thiếu cột, tránh KeyError.

---

### Hàm 3: `handle_missing_values(df)`

```python
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df
```

**Chiến lược điền missing:**

| Kiểu dữ liệu | Chiến lược | Lý do |
|---|---|---|
| Số (`float64`, `int64`) | Điền **median** | Robust với outliers (Total Charges có outlier lớn) |
| Categorical (`object`) | Điền **mode** (giá trị phổ biến nhất) | Giữ phân phối tự nhiên của dữ liệu |

**Tại sao nhiều missing?** Các cột service (Online Security, Streaming TV...) chỉ có giá trị với khách dùng Internet. Khách không dùng Internet → NaN. Đây là **structural missing**, không phải lỗi thu thập dữ liệu.

**FutureWarning:** `inplace=True` với `fillna()` trong pandas mới sẽ bị deprecate. Code vẫn chạy đúng, chỉ là cảnh báo cosmetic.

---

### Hàm 4: `encode_categorical(df)`

```python
def encode_categorical(df: pd.DataFrame) -> tuple:
    df = df.copy()
    encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders
```

**LabelEncoder hoạt động như thế nào?**

```
'Female' → 0,  'Male' → 1
'No'     → 0,  'Yes'  → 1
'Month-to-Month' → 0, 'One Year' → 1, 'Two Year' → 2
```

**Tại sao lưu `encoders`?** Để sau này decode ngược lại khi giải thích (LIME/SHAP cần tên feature gốc, không phải số).

**`.astype(str)` trước `fit_transform`:** Đảm bảo không bị lỗi nếu cột có giá trị mixed type (NaN lẫn string).

---

### Hàm 5: `scale_features(X_train, X_test)`

```python
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
```

**StandardScaler — công thức:**

$$z = \frac{x - \mu}{\sigma}$$

Trong đó $\mu$ và $\sigma$ được tính **chỉ trên tập train**, sau đó áp dụng lên cả test.

**Tại sao `fit_transform` cho train, chỉ `transform` cho test?**

Nếu `fit_transform` cả hai → **data leakage**: thông tin từ test set ảnh hưởng vào quá trình chuẩn hoá → metric giả tạo.

| Tập | Thao tác | Lý do |
|---|---|---|
| Train | `fit_transform()` | Tính $\mu$, $\sigma$ rồi scale |
| Test | `transform()` | Dùng $\mu$, $\sigma$ từ train để scale |

---

### Hàm 6: `preprocess_data(df, ...)` — Hàm Orchestrator Chính

```python
def preprocess_data(df, test_size=0.25, random_state=42, use_smote=False) -> dict:
    # Bước 1–4: filter, select, missing, encode
    df = create_target_variable(df)
    df = select_features(df)
    df = handle_missing_values(df)
    df, encoders = encode_categorical(df)

    # Bước 5: tách X, y
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Bước 6: Split 75/25
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Bước 7: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Bước 8: SMOTE (tắt)
    if use_smote: ...  # Không chạy

    return {
        'X_train': X_train_scaled, 'X_test': X_test_scaled,
        'y_train': ..., 'y_test': ...,
        'feature_names': feature_names,
        'encoders': encoders, 'scaler': scaler,
        'X_train_df': X_train, 'X_test_df': X_test
    }
```

**Tại sao `stratify=y`?**

Đảm bảo tỉ lệ churn trong train/test giống nhau (28.4%). Không stratify → có thể test set có churn rate khác nhiều → metric không đáng tin.

**Tại sao `use_smote=False`?**

Bài báo không đề cập SMOTE. Thay vào đó dùng `class_weight='balanced'` trong model (xem `models.py`) — cách này không tạo dữ liệu giả, mà điều chỉnh loss weight cho mỗi class.

**Số liệu thực tế sau split:**

| | Số lượng | Churn rate |
|---|---|---|
| Train | 4941 | 28.4% |
| Test | 1648 | 28.4% |
| **Tổng** | **6589** | **28.4%** |

**Dict trả về:**

| Key | Kiểu | Ý nghĩa |
|---|---|---|
| `X_train` | `np.ndarray` (4941×25) | Feature train đã scale |
| `X_test` | `np.ndarray` (1648×25) | Feature test đã scale |
| `y_train` | `np.ndarray` (4941,) | Nhãn train |
| `y_test` | `np.ndarray` (1648,) | Nhãn test |
| `feature_names` | `list` (25 tên) | Tên feature cho explainability |
| `encoders` | `dict` | LabelEncoder của từng cột categorical |
| `scaler` | `StandardScaler` | Để inverse transform sau này |
| `X_train_df` | `pd.DataFrame` | DataFrame gốc chưa scale (cho LIME) |
| `X_test_df` | `pd.DataFrame` | DataFrame gốc chưa scale (cho LIME) |

---

## Luồng dữ liệu tóm tắt

```
DataFrame(7043×38)
    ↓ create_target_variable()  → Lọc "Joined" → 6589 hàng, thêm cột 'Churn'
    ↓ select_features()         → Chọn 25 feature → 6589×26 (25 + Churn)
    ↓ handle_missing_values()   → Điền median/mode → 0 missing
    ↓ encode_categorical()      → Object → int → sẵn sàng cho sklearn
    ↓ train_test_split()        → Train(4941), Test(1648)
    ↓ scale_features()          → μ=0, σ=1 cho từng feature

Output: X_train(4941×25), X_test(1648×25), y_train(4941,), y_test(1648,)
```
