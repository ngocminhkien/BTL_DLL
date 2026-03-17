# Dự đoán trả hàng TMĐT (E-commerce Returns Prediction)

## 1) Giới thiệu dự án

Dự án tập trung vào việc phân tích hành vi khách hàng và đặc điểm sản phẩm để dự đoán khả năng hoàn trả đơn hàng trong TMĐT. Quy trình tuân theo pipeline:

**Nguồn dữ liệu → Tiền xử lý → Đặc trưng → Mô hình hóa → Đánh giá**

Mục tiêu:

- Dự đoán biến mục tiêu `is_return` (phân lớp).
- Khai phá luật kết hợp và phân cụm khách hàng để rút ra insight hành động giúp giảm hoàn hàng.

## 2) Cấu trúc Repository

```text
DATA_MINING_PROJECT/
├── configs/
│   └── params.yaml                 # Tham số: seed, split, paths, hyperparams
├── data/
│   ├── raw/                        # Dữ liệu gốc
│   └── processed/                  # Dữ liệu sau tiền xử lý
├── notebooks/                      # Notebook chạy theo thứ tự
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation.ipynb         # Evaluation + actionable insights + ARIMA forecast
├── src/                            # Logic chính
│   ├── data/                       # Loader/Cleaner
│   ├── features/                   # Feature engineering (RFM, bins)
│   ├── mining/                     # Association rules & clustering
│   └── models/                     # Supervised & Semi-supervised
├── scripts/
│   ├── run_pipeline.py             # Chạy pipeline tự động + chạy notebook 01→05
│   └── normalize_notebook.py       # Chuẩn hóa notebook để chạy tái lập
├── outputs/                        # Kết quả đầu ra (không commit)
│   ├── figures/                    # Biểu đồ (Confusion Matrix, forecast, ...)
│   ├── models/                     # Model đã huấn luyện (.pkl)
│   ├── metrics/                    # Metrics + forecast (.csv/.json)
│   └── notebooks/                  # Notebook đã execute (bản reproducible)
├── requirements.txt                # Thư viện phụ thuộc
└── README.md
```

## 3) Hướng dẫn cài đặt và thực thi (Reproducibility)

### Bước 1: Cài đặt môi trường

Yêu cầu: Python 3.9+.

```bash
python -m pip install -r requirements.txt
```

### Bước 2: Chuẩn bị dữ liệu
Tải bộ dữ liệu tại:https://www.kaggle.com/datasets/carrie1/ecommerce-data

Đặt file `data.csv` vào:

```text
data/raw/data.csv
```

### Bước 3: Cấu hình tham số

Chỉnh tham số tại `configs/params.yaml` (nếu cần).

### Bước 4: Chạy pipeline

```bash
python scripts/run_pipeline.py
```

Script sẽ:

- Làm sạch dữ liệu + tạo features
- Train model + lưu `.pkl`
- Lưu metrics/figures vào `outputs/`
- Chạy notebook 01→05 và lưu bản đã chạy vào `outputs/notebooks/`

## 4) Kết quả đạt được

Sau khi chạy, kết quả được lưu trong `outputs/`:

- **Figures**: phân phối, Confusion Matrix, learning curve (semi-supervised), ARIMA forecast.
- **Models**: `best_model.pkl` và các model theo timestamp.
- **Metrics/Reports**: bảng metric (F1, ROC-AUC, PR-AUC), JSON chi tiết, dự báo ARIMA theo tháng.

## 5) Insight hành động (Actionable Insights)

Chi tiết các insight kèm hành động nằm trong `notebooks/05_evaluation.ipynb`, ví dụ:

- Kiểm soát quy trình đóng gói cho **Top sản phẩm có tỉ lệ trả cao** (xuất từ `outputs/metrics/top10_risky_products.csv`).
- Tối ưu chính sách theo phân khúc khách hàng (VIP vs Return-prone).
- Tuning threshold để giảm False Negative tùy mục tiêu kinh doanh.
