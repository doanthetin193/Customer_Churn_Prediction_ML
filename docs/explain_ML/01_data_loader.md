# Giải thích: `data_loader.py` — Bước 1: Tải Dữ Liệu

## Vị trí trong Pipeline

```
main.py
 └─► [BƯỚC 1] data_loader.py  ← Bạn đang ở đây
      └─► data_preprocessing.py
           └─► models.py
                └─► evaluation.py
                     └─► explainability.py
```

---

## Mục đích

File này thực hiện đúng 1 nhiệm vụ duy nhất: **đọc file CSV dataset vào bộ nhớ** và in ra thống kê cơ bản để kiểm tra dữ liệu đầu vào trước khi xử lý.

---

## Liên hệ với Bài Báo (Section 3.2 — Dataset)

Bài báo (Chang et al., 2024) mô tả dataset:

> *"A Telecom CUSTOMER Churn dataset from the Maven Analytics website platform was selected. It consists of customer activity data (features), along with a churn label specifying whether a customer canceled the subscription. The dataset consists of 7043 rows and 38 attributes."*

| Thông tin trong bài báo | Tương ứng trong code |
|---|---|
| Dataset: Maven Analytics Telecom | `telecom_customer_churn.csv` |
| 7043 dòng, 38 thuộc tính | In ra qua `df.shape` |
| Biến target: Customer Status | `df['Customer Status'].value_counts()` |

---

## Phân tích Code Chi Tiết

### Hàm `load_data(data_path=None)`

```python
def load_data(data_path: str = None) -> pd.DataFrame:
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'dataset', 'telecom_customer_churn.csv'
        )
    df = pd.read_csv(data_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
```

**Giải thích từng dòng:**

| Dòng | Ý nghĩa |
|---|---|
| `data_path: str = None` | Tham số tuỳ chọn — gọi `load_data()` không cần argument vẫn chạy được |
| `os.path.dirname(__file__)` | Lấy thư mục chứa file `data_loader.py` (= `main/`) |
| `'..', 'dataset', '...'` | Đi lên 1 cấp (`main/` → project root) rồi vào `dataset/` |
| `pd.read_csv(data_path)` | Đọc CSV thành DataFrame — cấu trúc bảng 2D của pandas |
| `df.shape` | Tuple `(số_hàng, số_cột)` → dùng để xác nhận load đúng |
| `-> pd.DataFrame` | Type hint: hàm trả về DataFrame |

**Đường dẫn thực tế được resolve:**
```
main\.. → project root → dataset\telecom_customer_churn.csv
```

Dùng `os.path.join` + `..` thay vì hardcode đường dẫn tuyệt đối — code chạy đúng trên mọi máy, mọi OS.

---

### Hàm `explore_data(df)`

```python
def explore_data(df: pd.DataFrame) -> dict:
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'target_distribution': df['Customer Status'].value_counts().to_dict()
    }

    print(f"Total samples: {info['shape'][0]}")
    print(f"Total features: {info['shape'][1]}")
    print(f"Missing values: {info['missing_values']}")
    print(f"\nTarget Distribution (Customer Status):")
    for status, count in info['target_distribution'].items():
        pct = count / info['shape'][0] * 100
        print(f"  - {status}: {count} ({pct:.1f}%)")

    return info
```

**Từng key trong dict `info`:**

| Key | Giá trị thực tế khi chạy | Ý nghĩa |
|---|---|---|
| `shape` | `(7043, 38)` | Kích thước dataset gốc |
| `columns` | Danh sách 38 tên cột | Để kiểm tra đặt tên đúng |
| `dtypes` | `{float64: 13, object: 25}` | Biết cột nào cần encode |
| `missing_values` | `30849` | **Rất nhiều!** — cần xử lý ở bước 2 |
| `target_distribution` | `{Stayed: 4720, Churned: 1869, Joined: 454}` | Phân phối nhãn |

**Phân phối nhãn thực tế:**

```
Stayed  : 4720 (67.0%)  ← Không rời bỏ
Churned : 1869 (26.5%)  ← Rời bỏ
Joined  :  454  (6.4%)  ← Khách hàng mới gia nhập
```

Đây là dataset **mất cân bằng** (imbalanced) — bài báo cũng nhận xét điều này qua Figure 3:
> *"It can be observed from the chart that the dataset is unbalanced, with almost twice as many churning samples than not churning samples."*

**Lưu ý:** `df.isnull().sum().sum()` dùng `.sum()` hai lần:
- Lần 1: Tổng missing theo từng cột → Series
- Lần 2: Tổng tất cả cột lại → scalar

---

## Output thực tế khi chạy

```
✓ Dataset loaded: 7043 rows, 38 columns

==================================================
DATASET OVERVIEW
==================================================
Total samples: 7043
Total features: 38
Missing values: 30849

Target Distribution (Customer Status):
  - Stayed: 4720 (67.0%)
  - Churned: 1869 (26.5%)
  - Joined: 454 (6.4%)
```

---

## Tại sao Missing Values nhiều vậy? (30.849 / 267.634 ô)

Dataset có nhiều cột service (Online Security, Online Backup, v.v.) chỉ điền giá trị cho khách hàng có Internet Service. Khách không dùng Internet → các cột đó để trống. Đây là **missing theo cấu trúc** (structural missing), không phải lỗi dữ liệu — sẽ được xử lý ở `data_preprocessing.py`.

---

## Dữ liệu chuyển tiếp sang bước tiếp theo

```python
# Trong main.py
df = load_data()          # DataFrame 7043×38
explore_data(df)          # In thống kê, trả về dict info
data = preprocess_data(df)  # Truyền df vào bước 2
```

`load_data()` trả về DataFrame **thô**, chưa xử lý gì — đây là nguyên tắc single responsibility: mỗi hàm làm đúng 1 việc.
