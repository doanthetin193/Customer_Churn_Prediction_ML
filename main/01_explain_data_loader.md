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

File này thực hiện đúng 1 nhiệm vụ duy nhất: **đọc file CSV dataset vào bộ nhớ** và in ra thống kê cơ bản để kiểm tra dữ liệu.

---

## Liên hệ với Bài Báo (Section 3.2 — Dataset)

Bài báo (Chang et al., 2024) mô tả dataset như sau:

> *"A Telecom CUSTOMER Churn dataset from the Maven Analytics website platform was selected. It consists of customer activity data (features), along with a churn label specifying whether a customer canceled the subscription. The dataset consists of 7043 rows and 38 attributes."*

Code trong file này nạp chính xác dataset đó:

| Thông tin trong bài báo | Tương ứng trong code |
|---|---|
| Dataset: Maven Analytics Telecom | `telecom_customer_churn.csv` |
| 7043 dòng, 38 thuộc tính | Được in ra: `df.shape` |
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
| `data_path = None` | Tham số có giá trị mặc định, cho phép gọi `load_data()` không cần argument |
| `os.path.dirname(__file__)` | Lấy thư mục chứa file `data_loader.py` (= `main/`) |
| `'..', 'dataset', '...'` | Đi lên 1 cấp (`main/ → project root`) rồi vào `dataset/` |
| `pd.read_csv(data_path)` | Đọc file CSV thành DataFrame — cấu trúc bảng 2D của pandas |
| `df.shape` | Trả về tuple `(số_hàng, số_cột)` |
| `-> pd.DataFrame` | Type hint: hàm này trả về một DataFrame |

**Đường dẫn thực tế được tạo ra:**
```
D:\Customer_Churn_Prediction_ML\main\..\dataset\telecom_customer_churn.csv
→ D:\Customer_Churn_Prediction_ML\dataset\telecom_customer_churn.csv
```

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
    ...
    return info
```

**Mục đích:** In ra thống kê sơ bộ để **xác nhận data load đúng** trước khi xử lý.

| Thông tin thu thập | Lý do |
|---|---|
| `shape` | Kiểm tra: đúng 7043 dòng, 38 cột chưa? |
| `columns` | Xem danh sách tên các cột thuộc tính |
| `dtypes` | Phân loại: cột nào là số, cột nào là text |
| `missing_values` | Dataset thực tế có **30849 giá trị thiếu** (nhiều cột dịch vụ = NaN khi khách không đăng ký) → xử lý ở bước sau |
| `target_distribution` | Phân bố Churned / Stayed / Joined — quan trọng để biết data có mất cân bằng không |

**Output thực tế khi chạy:**
```
==================================================
DATASET OVERVIEW
==================================================
Total samples: 7043
Total features: 38
Missing values: 30849

Target Distribution (Customer Status):
  - Stayed: 4720 (67.0%)
  - Churned: 1869 (26.5%)
  - Joined: 454 (6.4%)    ← Nhóm này sẽ bị gán nhãn 0 ở bước preprocessing
```

> **Lưu ý quan trọng:** Missing values = **30849** — con số lớn vì nhiều cột dịch vụ bổ sung (Online Security, Streaming TV...) có giá trị `NaN` khi khách hàng không dùng Internet. Bướcpreprocessing sẽ điền các giá trị này.

---

## Lưu Đồ Luồng Dữ Liệu

```
[ổ đĩa] 
  telecom_customer_churn.csv (7043 dòng × 38 cột, dạng text)
          │
          │  pd.read_csv()
          ▼
[RAM / bộ nhớ]
  df: DataFrame (7043 × 38)
  - Cột Customer Status: "Churned" / "Stayed" / "Joined"
  - Cột số: Age, Tenure in Months, Monthly Charge, ...
  - Cột text: Gender, Contract, Internet Type, ...
          │
          │  explore_data()
          ▼
[Console]
  In thống kê cơ bản → xác nhận data OK
          │
          ▼
[Trả về df] → đưa sang data_preprocessing.py
```

---

## Điểm Quan Trọng Cần Nhớ

1. **Dataset gốc có 3 nhóm Customer Status:** `Churned`, `Stayed`, `Joined`
   - Bài báo chỉ quan tâm đến **Churned vs Not Churned**
   - Nhóm `Joined` (khách hàng mới) sẽ được xử lý ở bước preprocessing

2. **File này không thay đổi dữ liệu** — chỉ đọc và báo cáo. Mọi biến đổi đều xảy ra ở `data_preprocessing.py`

3. **Kiểu trả về là `pd.DataFrame`** — đây là đối tượng trung tâm của toàn bộ pipeline

---

## Cách Chạy Độc Lập (để test)

```bash
cd D:\Customer_Churn_Prediction_ML
.venv\Scripts\activate
cd main
python data_loader.py
```

---

*Tiếp theo: [02_explain_data_preprocessing.md](02_explain_data_preprocessing.md)*
