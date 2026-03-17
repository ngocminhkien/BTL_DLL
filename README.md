# Dự đoán trả hàng TMĐT với Online Retail

1. Giới thiệu dự án
Dự án tập trung vào việc phân tích hành vi khách hàng và đặc điểm sản phẩm để dự đoán khả năng hoàn trả đơn hàng trong TMĐT. Quy trình thực hiện tuân theo pipeline: Nguồn dữ liệu -> Tiền xử lý -> Đặc trưng -> Mô hình hóa -> Đánh giá.

2. Cấu trúc Repository 

Dự án được tổ chức theo cấu trúc module hóa chuyên nghiệp:

Plaintext
DATA_MINING_PROJECT/
├── configs/
│   └── params.yaml          # Quản lý tham số: seed, split, paths, hyperparams [cite: 82, 149]
├── data/
│   ├── raw/                 # Dữ liệu gốc (không commit lên GitHub) [cite: 84, 151]
│   └── processed/           # Dữ liệu sau tiền xử lý [cite: 85]
├── notebooks/               # Chứa pipeline thực thi theo thứ tự [cite: 147]
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_or_clustering.ipynb
│   ├── 04_modeling.ipynb
│   ├── 04b_semi_supervised.ipynb
│   └── 05_evaluation_report.ipynb
├── src/                     # Toàn bộ logic chính của dự án [cite: 96, 148]
│   ├── data/                # Module loader, cleaner [cite: 98]
│   ├── features/            # Module xây dựng đặc trưng (RFM, Bins) [cite: 102]
│   ├── mining/              # Module luật kết hợp và phân cụm [cite: 109]
│   ├── models/              # Module Supervised & Semi-supervised [cite: 115]
│   └── evaluation/          # Module tính toán metrics & visualization [cite: 125]
├── scripts/
│   └── run_pipeline.py      # Script chạy toàn bộ pipeline tự động [cite: 134, 143]
├── outputs/                 # Kết quả đầu ra của dự án [cite: 135]
│   ├── figures/             # Các biểu đồ, hình ảnh báo cáo [cite: 136]
│   ├── models/              # Các file mô hình đã huấn luyện (.pkl) 
│   └── reports/             # Bảng biểu và kết quả tổng hợp [cite: 139]
├── requirements.txt         # Danh sách thư viện cần thiết [cite: 79, 157]
└── README.md
3. Hướng dẫn cài đặt và thực thi (Reproducibility) 

Bước 1: Cài đặt môi trường
Đảm bảo bạn đã cài đặt Python 3.9+ và chạy lệnh sau để cài đặt thư viện:

Bash
pip install -r requirements.txt [cite: 157]
Bước 2: Chuẩn bị dữ liệu 

Tải bộ dữ liệu tại: Kaggle E-commerce Returns Dataset.

Giải nén và đặt file data.csv vào thư mục data/raw/.

Bước 3: Cấu hình tham số
Cập nhật các đường dẫn và tham số huấn luyện (nếu cần) tại file configs/params.yaml.

Bước 4: Chạy Pipeline 

Bạn có thể chạy lần lượt các Notebook trong thư mục notebooks/ hoặc chạy script tự động để sinh ra toàn bộ Artifacts (kết quả):

Bash
python scripts/run_pipeline.py 
4. Kết quả đạt được 

Sau khi thực thi, các kết quả sau sẽ được tự động lưu vào thư mục outputs/:

Figures: Biểu đồ phân phối, Learning Curve cho Semi-supervised, Confusion Matrix.


Models: Mô hình tốt nhất đã được lưu dưới dạng .pkl.

Reports: Bảng so sánh metric (F1, PR-AUC) giữa Baseline và XGBoost.

5. Insight hành động (Actionable Insights)
Dự án cung cấp ít nhất 5 kiến nghị cụ thể cho doanh nghiệp TMĐT nhằm giảm tỉ lệ trả hàng, bao gồm:

Kiểm soát quy trình đóng gói nhóm sản phẩm có tỉ lệ trả hàng cao.

Tối ưu hóa chính sách đổi trả tại các thị trường trọng điểm (như Ireland).
... (Chi tiết xem tại notebooks/05_evaluation_report.ipynb)
