# Giải thích: `data_preprocessing.py` — Bước 2: Tiền Xử Lý Dữ Liệu

## Vị trí trong Pipeline

```
main.py
 └─► data_loader.py  [✓ Xong]
      └─► [BƯỚC 2] data_preprocessing.py  ← Bạn đang ở đây
           └─► models.py
                └─► evaluation.py
                     └─► explainability.py
```

---

## Tổng Quan

File này thực hiện toàn bộ **chuỗi xử lý dữ liệu thô** trước khi đưa vào huấn luyện mô hình. Bài báo không đề cập chi tiết từng bước code, nhưng đây là quy trình chuẩn trong ML được ngầm định trong Section 3.2 và 3.3.

**Đầu vào:** `df` — DataFrame thô (7043 × 38 cột)  
**Đầu ra:** Dictionary chứa `X_train`, `X_test`, `y_train`, `y_test` dạng numpy array

---

## Liên Hệ Bài Báo

| Bài báo nói | Code thực hiện |
|---|---|
| *"The research data was divided into Train (75%) and Test (25%)"* | `train_test_split(..., test_size=0.25)` |
| *"the great ratio of 2:1 (non-churner: churner)"* | `stratify=y` → giữ tỷ lệ này khi split |
| *"none of the observations or rows were deleted"* | `handle_missing_values()` điền thay vì xóa |
| *"relevant explanatory variables such as demographics"* | `select_features()` chọn 25 cột |
| Dataset mất cân bằng (Fig 3 trong bài báo) | Code thêm SMOTE (ngoài bài báo) để cải thiện |

---

## Phân Tích Từng Hàm

### 1. `create_target_variable(df)` — Tạo nhãn nhị phân

```python
def create_target_variable(df):
    df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)
    return df
```

**Tại sao cần bước này?**

Dataset gốc có cột `Customer Status` với 3 giá trị:
```
Churned  → 1869 khách (26.5%)
Stayed   → 2961 khách (42.1%)  
Joined   → 2175 khách (30.9%)  ← khách mới gia nhập
```

Bài báo xử lý bài toán **phân loại nhị phân** (churn hoặc không churn), vì vậy:
- `'Churned'` → **1** (tích cực: đã rời bỏ)
- `'Stayed'` hoặc `'Joined'` → **0** (tiêu cực: không rời bỏ)

```
"Churned" == "Churned" → True  → .astype(int) → 1
"Stayed"  == "Churned" → False → .astype(int) → 0
"Joined"  == "Churned" → False → .astype(int) → 0
```

---

### 1b. Lưu ý: FutureWarning khi chạy

Khi chạy thực tế sẽ xuất hiện warning (không phải lỗi):

```
FutureWarning: A value is trying to be set on a copy of a DataFrame or Series
through chained assignment using an inplace method...
  df[col].fillna(df[col].mode()[0], inplace=True)
```

Đây là cảnh báo tương thích của pandas 3.0 — code vẫn chạy đúng, chỉ cần đổi sang `df[col] = df[col].fillna(...)` nếu muốn loại warning.

---

### 2. `select_features(df)` — Chọn 25 đặc trưng

```python
feature_cols = [
    'Gender', 'Age', 'Married', 'Number of Dependents',
    'Number of Referrals', 'Tenure in Months', 'Offer',
    'Phone Service', 'Multiple Lines', 'Internet Service', 'Internet Type',
    'Online Security', 'Online Backup', 'Device Protection Plan',
    'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
    'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
    'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Revenue'
]
```

**Nhóm theo ý nghĩa:**

| Nhóm | Cột | Bài báo đề cập |
|---|---|---|
| **Nhân khẩu học** | Gender, Age, Married, Number of Dependents | *"demographics"* |
| **Hành vi dùng dịch vụ** | Tenure, Referrals, Offer, Phone/Internet Service | Section 3.2 |
| **Dịch vụ bổ sung** | Online Security, Backup, Streaming TV/Movies/Music... | Section 3.2 |
| **Hợp đồng & thanh toán** | Contract, Paperless Billing, Payment Method | Section 3.2 |
| **Tài chính** | Monthly Charge, Total Charges, Total Revenue | Table 2 (Descriptive statistics) |

Dataset gốc có 38 cột, code loại bỏ các cột không cần thiết như `Customer ID`, `City`, `Zip Code`, `Latitude`, `Longitude`, `Avg Monthly GB Download`...

> **Lưu ý:** Dataset thực tế có **30849 giá trị thiếu** (NaN) vì nhiều cột dịch vụ bổ sung rỗng khi khách hàng không dùng Internet.

---

### 3. `handle_missing_values(df)` — Xử lý giá trị thiếu

```python
# Cột số → điền bằng MEDIAN (trung vị)
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Cột text → điền bằng MODE (giá trị xuất hiện nhiều nhất)
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
```

**Tại sao dùng median cho số, mode cho text?**

- **Median** ít bị ảnh hưởng bởi outlier hơn mean (ví dụ: Total Charges có giá trị từ 42 đến 8684)
- **Mode** là lựa chọn an toàn cho biến phân loại (không có khái niệm trung bình của "Yes/No")

---

### 4. `encode_categorical(df)` — Mã hóa biến phân loại

```python
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
```

Các thuật toán ML (Logistic Regression, Random Forest...) **không hiểu text**, chúng chỉ làm việc với số. `LabelEncoder` chuyển đổi:

```
'Yes'  → 1        'No'  → 0
'Male' → 1        'Female' → 0
'Month-to-Month' → 0    'One Year' → 1    'Two Year' → 2
'DSL' → 0    'Fiber Optic' → 1    'Cable' → 2
```

> **Lưu ý:** `LabelEncoder` tạo thứ tự ngầm định (0 < 1 < 2). Với biến danh nghĩa không có thứ tự như `Internet Type`, phương pháp tốt hơn là `OneHotEncoding`, nhưng bài báo không đề cập chi tiết nên code dùng cách đơn giản này. Thực tế khi chạy encode được **18 cột categorical**.

---

### 5. `scale_features()` — Chuẩn hóa dữ liệu

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Học mean/std từ train rồi transform
X_test_scaled = scaler.transform(X_test)         # Chỉ transform (dùng mean/std của train)
```

**Công thức StandardScaler:**

$$z = \frac{x - \mu}{\sigma}$$

Sau khi scale, mỗi feature có **mean = 0** và **std = 1**.

**Tại sao cần scale?**

| Thuật toán | Ảnh hưởng khi không scale |
|---|---|
| Logistic Regression | Hội tụ chậm, kết quả kém |
| KNN | Bị bias bởi feature có giá trị lớn (ví dụ Total Revenue 0–12000 sẽ áp đảo Age 19–80) |
| Naive Bayes | Ít ảnh hưởng |
| Decision Tree / Random Forest | **Không cần scale** (tree-based không bị ảnh hưởng bởi scale), nhưng không gây hại |

**Quan trọng:**  `fit_transform` chỉ áp dụng trên **train set** để tránh data leakage — test set chỉ được `transform` bằng thống kê từ train.

---

### 6. SMOTE — Xử lý Mất Cân Bằng (Ngoài Bài Báo)

```python
if use_smote:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
```

Bài báo xác nhận dataset **mất cân bằng** (Figure 3):

```
Churned: 1869 (26.5%)  ████
Stayed:  4720 (67.0%)  █████████████████████████
Joined:   454  (6.4%)  ██
```

Sau khi tạo target binary (Churned=1, phần còn lại=0):
```
Churn (1):     26.5%  ████
Not Churn (0): 73.5%  ████████████████████████
```

SMOTE (Synthetic Minority Over-sampling Technique) **tạo ra mẫu tổng hợp** cho class thiểu số (Churned) bằng cách nội suy giữa các điểm dữ liệu lân cận, thay vì chỉ sao chép. Nhờ đó:
- Train set trở nên cân bằng ~50/50
- Model không bị thiên vị về class "Not Churn"
- Sensitivity (recall cho Churned) tăng đáng kể

> SMOTE chỉ áp dụng cho **train set**, không dùng cho test set.

---

### 7. `preprocess_data()` — Pipeline Tổng Hợp

```python
def preprocess_data(df, test_size=0.25, random_state=42, use_smote=False):
    df = create_target_variable(df)   # Bước 1
    df = select_features(df)          # Bước 2
    df = handle_missing_values(df)    # Bước 3
    df, encoders = encode_categorical(df)  # Bước 4
    
    X = df.drop('Churn', axis=1)   # 25 features
    y = df['Churn']                # target (0/1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y  # Bước 5
    )
    
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)  # Bước 6
    
    if use_smote: ...  # Bước 7 (tùy chọn)
    
    return {'X_train': ..., 'X_test': ..., 'y_train': ..., 'y_test': ..., ...}
```

---

## Sơ Đồ Biến Đổi Dữ Liệu

```
DataFrame gốc (7043 × 38)
        │
        │ create_target_variable()
        ▼
+ cột 'Churn' (0/1)  → Customer Status = 'Churned' ? 1 : 0
        │
        │ select_features()
        ▼
DataFrame (7043 × 26)   ← 25 features + 1 target
        │
        │ handle_missing_values()  [FutureWarning xuất hiện, không ảnh hưởng]
        ▼
Không còn NaN  (xử lý 30849 giá trị thiếu)
        │
        │ encode_categorical()  → 18 cột categorical
        ▼
Toàn bộ là số (int/float)
        │
        │ train_test_split(test_size=0.25, stratify=y)
        ▼
   X_train (5282 × 25)   X_test (1761 × 25)
   y_train (5282,)        y_test (1761,)
        │
        │ StandardScaler.fit_transform()
        ▼
   X_train_scaled         X_test_scaled
   (mean=0, std=1)        (dùng stats của train)
        │
        │ SMOTE (use_smote=True) ← mặc định bật trong main.py
        ▼
   X_train_scaled (7760 × 25)  ← tăng từ 5282 lên 7760
   y_train (7760,)              ← Churn rate = 50.0% (cân bằng)
```

---

## Đầu Ra (Dictionary)

```python
{
    'X_train': np.ndarray,   # Features train (đã scale, có thể đã SMOTE)
    'X_test':  np.ndarray,   # Features test (đã scale)
    'y_train': np.ndarray,   # Labels train (0/1)
    'y_test':  np.ndarray,   # Labels test (0/1)
    'feature_names': list,   # ['Gender', 'Age', ..., 'Total Revenue']
    'encoders': dict,        # LabelEncoder cho từng cột categorìcal
    'scaler': StandardScaler,# Để transform dữ liệu mới sau này
    'X_train_df': DataFrame, # Bản gốc chưa scale (dùng cho LIME/SHAP)
    'X_test_df':  DataFrame  # Bản gốc chưa scale (dùng cho LIME/SHAP)
}
```

---

*Trước: [01_explain_data_loader.md](01_explain_data_loader.md)*  
*Tiếp theo: [03_explain_models.md](03_explain_models.md)*
