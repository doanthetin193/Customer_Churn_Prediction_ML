# Customer Churn Prediction — ML & Deep Learning with XAI

Dự án tiểu luận môn **Học Máy và Ứng Dụng** — Trường Đại học Quy Nhơn.

Triển khai và nâng cấp nghiên cứu của Chang et al. (2024), so sánh 5 mô hình học máy cổ điển (ML) và một pipeline học sâu MLP (DL), kết hợp giải thích mô hình bằng LIME và SHAP.

> Chang, V., Hall, K., Xu, Q. A., Amao, F. O., Ganatra, M. A., & Benson, V. (2024). Prediction of Customer Churn Behavior in the Telecommunication Industry Using Machine Learning Models. *Algorithms*, 17(6), 231. https://doi.org/10.3390/a17060231

---

## Tổng quan dự án

Dự án gồm hai pipeline độc lập:

| Pipeline | Thư mục | Mô hình | File chính |
|---|---|---|---|
| **ML** (tái lập bài báo) | `main/` | LR, KNN, NB, DT, Random Forest | `main/main.py` |
| **DL** (nâng cấp) | `main_DL/` | MLP 256→128→64 | `main_DL/main.py` |

**Dataset:** Telecom Customer Churn (Maven Analytics) — 7.043 dòng → 6.589 sau lọc "Joined".

**Giảng viên hướng dẫn:** TS. Lê Quang Hùng

---

## Kết quả thực nghiệm

### ML Pipeline — 5 mô hình học máy

| Mô hình | Accuracy | Sensitivity | Specificity | AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.46% | **86.72%** | 77.98% | 0.9112 |
| KNN | 81.01% | 66.81% | 86.62% | 0.8521 |
| Naïve Bayes | 80.10% | 78.59% | 80.69% | 0.8771 |
| Decision Tree | 82.10% | 68.09% | 87.64% | 0.7787 |
| **Random Forest** | **86.65%** | 74.52% | **91.45%** | **0.9264** |

**Random Forest** đạt kết quả tốt nhất (Accuracy 86.65%, AUC 0.9264), chênh lệch chỉ 0.29% so với bài báo gốc (86.94%).

### DL Pipeline — MLP (threshold = 0.54, Youden's J)

| Accuracy | Sensitivity | Specificity | AUC |
|---:|---:|---:|---:|
| 81.74% | **84.37%** | 80.69% | 0.9132 |

**MLP** đạt Sensitivity cao nhất trong tất cả 6 mô hình (**84.37%**) — bắt churner tốt hơn RF đến +9.85 điểm %.

---

## Cấu trúc dự án

```text
Customer_Churn_Prediction_ML/
├── dataset/
│   ├── telecom_customer_churn.csv       # Dataset chính (7043 khách hàng)
│   ├── telecom_data_dictionary.csv      # Mô tả các cột
│   └── telecom_zipcode_population.csv  # Dữ liệu dân số theo zipcode
│
├── main/                                # Pipeline ML (5 mô hình cổ điển + XAI)
│   ├── main.py                          # Entry point
│   ├── data_loader.py                   # Load & EDA
│   ├── data_preprocessing.py            # Tiền xử lý, split, scale
│   ├── models.py                        # 5 mô hình ML
│   ├── evaluation.py                    # Metrics, biểu đồ
│   ├── explainability.py                # LIME & SHAP
│   ├── requirements.txt
│   └── results/                         # Kết quả sau khi chạy
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       ├── metrics_comparison.png
│       ├── lime_sample_0.png
│       ├── lime_sample_1.png
│       ├── shap_summary.png
│       └── shap_bar.png
│
├── main_DL/                             # Pipeline DL (MLP học sâu)
│   ├── main.py                          # Entry point
│   ├── data_loader.py                   # Load & EDA
│   ├── data_preprocessing.py            # Tiền xử lý (dùng chung logic với ML)
│   ├── models_dl.py                     # Kiến trúc MLP + optimal threshold
│   ├── evaluation.py                    # Metrics, training history plot
│   ├── requirements.txt
│   └── results/
│       ├── training_history_dl.png
│       ├── confusion_matrix_dl.png
│       ├── roc_curve_dl.png
│       ├── metrics_dl.json
│       └── mlp_churn_model.keras
│
├── docs/                                # Tài liệu giải thích chi tiết
│   ├── ml_pipeline.md                   # Tổng quan pipeline ML
│   ├── dl_pipeline.md                   # Tổng quan pipeline DL
│   ├── explain_ML/                      # Giải thích từng file ML
│   └── explain_DL/                      # Giải thích từng file DL
│
├── BAO_CAO_TIEU_LUAN.md                # Báo cáo tiểu luận đầy đủ
├── requirements.txt                     # Dependencies tổng hợp
└── README.md
```

---

## Cài đặt

```bash
git clone https://github.com/doanthetin193/Customer_Churn_Prediction_ML.git
cd Customer_Churn_Prediction_ML

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Cách chạy

### Pipeline ML (5 mô hình cổ điển + LIME/SHAP)

```bash
python main/main.py
```

Pipeline thực hiện:
1. Load dataset và khám phá dữ liệu
2. Lọc khách hàng "Joined", chọn 25 features
3. Điền missing values, Label Encoding, StandardScaler
4. Chia train/test 75/25 (stratified)
5. Huấn luyện 5 mô hình với `class_weight='balanced'`
6. Đánh giá và vẽ confusion matrix, ROC curve, metrics comparison
7. Giải thích Random Forest bằng LIME (2 mẫu) và SHAP (100 mẫu)

Kết quả lưu tại `main/results/`.

### Pipeline DL (MLP học sâu)

```bash
python main_DL/main.py
```

Pipeline thực hiện:
1. Tiền xử lý tương tự ML pipeline
2. Chia thêm 20% validation từ train set
3. Huấn luyện MLP 3-block (256→128→64), max 150 epochs, EarlyStopping patience=20
4. Tìm threshold tối ưu bằng Youden's J trên validation set
5. Đánh giá trên test set, vẽ training history, confusion matrix, ROC
6. Lưu model `.keras`

Kết quả lưu tại `main_DL/results/`.

---

## Kiến trúc MLP (Pipeline DL)

```
Input (25 features)
  └── Dense(256) → BatchNorm → LeakyReLU(0.1) → Dropout(0.40)
        └── Dense(128) → BatchNorm → LeakyReLU(0.1) → Dropout(0.40)
              └── Dense(64)  → BatchNorm → LeakyReLU(0.1) → Dropout(0.20)
                    └── Dense(1) → Sigmoid → P(Churn)
```

- **Optimizer:** Adam, lr=5e-4
- **Regularization:** L2=3e-4
- **Class weight:** {Stayed: 0.698, Churned: 1.763}
- **Threshold:** 0.54 (Youden's J = 0.7472)

---

## Phương pháp xử lý mất cân bằng lớp

Không dùng SMOTE. Thay vào đó:
- **ML:** `class_weight='balanced'` trong scikit-learn
- **DL:** `compute_class_weight('balanced')` từ sklearn → truyền vào `model.fit(class_weight=...)`

Phương pháp này nhất quán với bài báo gốc Chang et al. (2024).

---

## Tài liệu

- [BAO_CAO_TIEU_LUAN.md](BAO_CAO_TIEU_LUAN.md) — Báo cáo tiểu luận đầy đủ (7 sections, kết quả + phân tích)
- [docs/ml_pipeline.md](docs/ml_pipeline.md) — Tổng quan kỹ thuật pipeline ML
- [docs/dl_pipeline.md](docs/dl_pipeline.md) — Tổng quan kỹ thuật pipeline DL
- [docs/explain_ML/](docs/explain_ML/) — Giải thích chi tiết từng file trong `main/`
- [docs/explain_DL/](docs/explain_DL/) — Giải thích chi tiết từng file trong `main_DL/`

---

## Tham khảo

1. Chang et al. (2024). *Algorithms*, 17(6), 231. https://doi.org/10.3390/a17060231
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. Ribeiro et al. (2016). LIME. *KDD 2016*, 1135–1144.
4. Lundberg & Lee (2017). SHAP. *NeurIPS 2017*.
5. Pedregosa et al. (2011). Scikit-learn. *JMLR*, 12, 2825–2830.
- [main/results/shap_bar.png](main/results/shap_bar.png)

## Tài liệu liên quan trong repo

- Báo cáo tiểu luận đã chuẩn hóa: [BAO_CAO_TIEU_LUAN.md](BAO_CAO_TIEU_LUAN.md)
- Mã nguồn thực nghiệm chính: [main/main.py](main/main.py), [main/models.py](main/models.py), [main/evaluation.py](main/evaluation.py), [main/explainability.py](main/explainability.py)

## Công nghệ sử dụng

- Python 3.10
- pandas, numpy
- scikit-learn
- imbalanced-learn
- matplotlib, seaborn
- lime, shap

## Ghi chú

- Dự án phục vụ học tập và nghiên cứu học thuật.
- Có thể mở rộng thêm nhánh thực nghiệm không SMOTE để đối chiếu với cấu hình paper baseline.
