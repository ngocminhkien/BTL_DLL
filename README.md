# Dự đoán trả hàng TMĐT với Online Retail

Project Bài tập lớn môn Khai phá dữ liệu (Data Mining) – Học kỳ II, năm học 2025-2026.

## Cấu trúc thư mục

- `configs/params.yaml`: Cấu hình đường dẫn, random seed, hyperparameters.
- `data/raw/`: Chứa file gốc `data.csv` (Online Retail).
- `data/processed/`: Lưu dữ liệu sau tiền xử lý và đặc trưng.
- `src/`: Các module Python chính
  - `cleaner.py`: Tiền xử lý dữ liệu, tạo biến `is_return`.
  - `features.py`: RFM, đặc trưng thời gian, vector hóa giỏ hàng.
  - `mining.py`: Luật kết hợp, phân cụm.
  - `models.py`: Mô hình phân lớp, bán giám sát, chuỗi thời gian.
  - `evaluation.py`: Tính toán metric, vẽ biểu đồ, SHAP.
- `notebooks/`: Các notebook `01_eda` đến `05_evaluation` gọi hàm từ `src/`.
- `scripts/run_pipeline.py`: Chạy toàn bộ pipeline từ đầu đến cuối.
- `outputs/`: Nơi lưu **kết quả chạy pipeline** (figures, models, metrics).

## Cài đặt môi trường

```bash
python -m pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

Đặt file `data.csv` (Online Retail) vào thư mục:

```text
data/raw/data.csv
```

## Chạy pipeline

```bash
python scripts/run_pipeline.py
```

Các kết quả chính (metric, hình vẽ, model `.pkl`) sẽ được lưu trong thư mục `outputs/`:

- `outputs/figures/`: biểu đồ (Confusion Matrix, …)
- `outputs/models/`: model `.pkl` (có `best_model.pkl`)
- `outputs/metrics/`: bảng metric `.csv` và báo cáo `.json`

