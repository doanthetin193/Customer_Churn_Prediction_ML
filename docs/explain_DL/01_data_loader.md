# Giải thích: `data_loader.py` — Bước 1: Tải Dữ Liệu (DL Pipeline)

## Vị trí trong Pipeline

```
main.py (DL)
 └─► [BƯỚC 1] data_loader.py  ← Bạn đang ở đây
      └─► data_preprocessing.py
           └─► models_dl.py
                └─► evaluation.py
```

---

## So sánh với ML Pipeline

`main_DL/data_loader.py` về cơ bản **giống** `main/data_loader.py`, nhưng có vài điểm khác biệt nhỏ:

| Điểm khác | ML (`main/data_loader.py`) | DL (`main_DL/data_loader.py`) |
|---|---|---|
| Type hint union syntax | `str = None` | `str \| None = None` (Python 3.10+ syntax) |
| `missing_values` cast | Không có | `int(df.isnull().sum().sum())` để serialize JSON |
| Style in/out | `✓` unicode | In thẳng không có prefix |
| Import style | `pd` only | `pd` only |

---

## Mục đích

Đọc file CSV dataset vào bộ nhớ và in thống kê cơ bản — **giống hệt** nhiệm vụ của ML pipeline. Cả hai pipeline cùng dùng chung một dataset nguồn.

---

## Liên hệ với Bài Báo (Section 3.2 — Dataset)

| Thông tin trong bài báo | Tương ứng trong code |
|---|---|
| Dataset: Maven Analytics Telecom | `telecom_customer_churn.csv` |
| 7043 dòng, 38 thuộc tính | In qua `df.shape` |
| Biến target: Customer Status | `df['Customer Status'].value_counts()` |

---

## Phân tích Code Chi Tiết

### Hàm 1: `load_data(data_path=None)`

```python
def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Load telecom churn dataset from CSV."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "dataset", "telecom_customer_churn.csv"
        )
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
```

**Giải thích từng dòng:**

| Dòng | Ý nghĩa |
|---|---|
| `str \| None = None` | Union type hint — Python 3.10+ syntax. Tương đương `Optional[str]` |
| `os.path.dirname(__file__)` | Thư mục chứa file → `main_DL/` |
| `'..', 'dataset', '...'` | Đi lên 1 cấp → project root → `dataset/` |
| `pd.read_csv(data_path)` | Đọc CSV → DataFrame |

**Đường dẫn resolve thực tế:**
```
main_DL\ → (lên 1 cấp) → project root → dataset\telecom_customer_churn.csv
```

---

### Hàm 2: `explore_data(df)`

```python
def explore_data(df: pd.DataFrame) -> dict:
    """Print and return quick dataset summary."""
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values": int(df.isnull().sum().sum()),  # cast to int
        "target_distribution": df["Customer Status"].value_counts().to_dict(),
    }

    print("\n" + "=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"Total samples: {info['shape'][0]}")
    print(f"Total features: {info['shape'][1]}")
    print(f"Missing values: {info['missing_values']}")
    print("\nTarget distribution (Customer Status):")
    for status, count in info["target_distribution"].items():
        pct = count / info["shape"][0] * 100
        print(f"  - {status}: {count} ({pct:.1f}%)")

    return info
```

**Tại sao có `int(df.isnull().sum().sum())`?**

`df.isnull().sum().sum()` trả về kiểu `numpy.int64` — không phải Python native `int`. Khi DL pipeline sau này serialize dict ra JSON (qua `json.dump()`), `numpy.int64` sẽ gây lỗi `TypeError: Object of type int64 is not JSON serializable`. Cast sang `int()` trước để phòng ngừa.

**Output thực tế:**

```
Loaded dataset: 7043 rows, 38 columns

==================================================
DATASET OVERVIEW
==================================================
Total samples: 7043
Total features: 38
Missing values: 30849

Target distribution (Customer Status):
  - Stayed: 4720 (67.0%)
  - Churned: 1869 (26.5%)
  - Joined: 454 (6.4%)
```

---

## Dữ liệu chuyển tiếp sang bước tiếp theo

```python
# Trong main.py (DL)
print("\nSTEP 1: Loading data")
df = load_data()
explore_data(df)

# ...

print("\nSTEP 2: Preprocessing data")
data = preprocess_data(df, use_smote=False)
```

`load_data()` trả về DataFrame **thô** 7043×38 — chưa filter "Joined", chưa scale, chưa encode. Toàn bộ biến đổi đó sẽ xảy ra ở `data_preprocessing.py`.
