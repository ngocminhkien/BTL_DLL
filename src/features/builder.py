import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _load_config(config_path: str) -> Tuple[dict, Path]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    project_root = cfg_path.resolve().parent.parent
    logger.info("Đã load cấu hình từ %s", cfg_path)
    return cfg, project_root


def load_clean_data(config_path: str = "configs/params.yaml") -> pd.DataFrame:
    """Tiện ích: đọc dữ liệu đã làm sạch từ đường dẫn trong cấu hình."""
    cfg, project_root = _load_config(config_path)
    paths_cfg = cfg.get("paths", {})
    cleaned_path = project_root / paths_cfg.get("cleaned_data", "data/processed/cleaned.csv")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu đã làm sạch: {cleaned_path}")

    logger.info("Đang đọc dữ liệu đã làm sạch từ %s", cleaned_path)
    df = pd.read_csv(
        cleaned_path,
        parse_dates=["InvoiceDate"],
        dtype={"InvoiceNo": str, "StockCode": str},
        encoding="ISO-8859-1",
    )
    return df


def build_rfm_features(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
) -> pd.DataFrame:
    """Tính toán đặc trưng RFM + ReturnRate theo CustomerID.

    - Recency: số ngày từ lần mua gần nhất đến ngày snapshot.
    - Frequency: số hóa đơn (InvoiceNo) khác nhau.
    - Monetary: tổng giá trị mua (Quantity * UnitPrice).
    - return_rate_customer: tỷ lệ dòng giao dịch trả hàng (is_return=1).
    """
    if df is None:
        df = load_clean_data(config_path)
    else:
        df = df.copy()

    if "CustomerID" not in df.columns:
        raise KeyError("Thiếu cột CustomerID trong dữ liệu đầu vào để tính RFM.")
    if "InvoiceDate" not in df.columns:
        raise KeyError("Thiếu cột InvoiceDate trong dữ liệu đầu vào để tính Recency.")
    if "InvoiceNo" not in df.columns:
        raise KeyError("Thiếu cột InvoiceNo trong dữ liệu đầu vào để tính Frequency.")
    if "Quantity" not in df.columns or "UnitPrice" not in df.columns:
        raise KeyError("Thiếu cột Quantity hoặc UnitPrice để tính Monetary.")

    cfg, _ = _load_config(config_path)
    rfm_cfg = cfg.get("rfm", {})
    snapshot_date = pd.to_datetime(rfm_cfg.get("recency_snapshot", "2011-12-10"))

    logger.info("Đang tính toán RFM cho từng khách hàng (CustomerID)...")

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    rfm = (
        df.groupby("CustomerID")
        .agg(
            last_purchase_date=("InvoiceDate", "max"),
            frequency=("InvoiceNo", "nunique"),
            monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    # Đảm bảo last_purchase_date là datetime để tránh lỗi khi trừ với snapshot_date
    rfm["last_purchase_date"] = pd.to_datetime(
        rfm["last_purchase_date"], errors="coerce"
    )
    rfm["recency"] = (snapshot_date - rfm["last_purchase_date"]).dt.days
    rfm.drop(columns=["last_purchase_date"], inplace=True)

    if "is_return" in df.columns:
        cust_return_rate = (
            df.groupby("CustomerID")["is_return"]
            .mean()
            .rename("return_rate_customer")
            .reset_index()
        )
        rfm = rfm.merge(cust_return_rate, on="CustomerID", how="left")
    else:
        rfm["return_rate_customer"] = np.nan

    logger.info("Đã tạo bảng RFM cho %d khách hàng.", len(rfm))
    return rfm


def compute_product_return_rate(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
) -> pd.DataFrame:
    """Tính tỷ lệ trả hàng cho từng sản phẩm (theo StockCode).

    Kết quả: DataFrame với index là StockCode và cột `product_return_rate`.
    """
    if df is None:
        df = load_clean_data(config_path)
    else:
        df = df.copy()

    if "StockCode" not in df.columns:
        raise KeyError("Thiếu cột StockCode trong dữ liệu để tính ReturnRate theo sản phẩm.")
    if "is_return" not in df.columns:
        raise KeyError("Thiếu cột is_return trong dữ liệu để tính ReturnRate.")

    logger.info("Đang tính tỷ lệ trả hàng cho từng sản phẩm (StockCode)...")
    prod_return = (
        df.groupby("StockCode")["is_return"]
        .mean()
        .rename("product_return_rate")
        .reset_index()
    )
    logger.info("Đã tạo bảng product_return_rate cho %d sản phẩm.", len(prod_return))
    return prod_return


def discretize_features(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
    n_bins: int = 4,
) -> pd.DataFrame:
    """Rời rạc hóa các biến số (UnitPrice, Quantity) thành các bins phân loại.

    - Tạo các cột mới: UnitPrice_bin, Quantity_bin.
    - Sử dụng qcut để chia theo quantile (mặc định 4 bins: thấp, trung bình thấp,
      trung bình cao, cao).
    """
    if df is None:
        df = load_clean_data(config_path)
    else:
        df = df.copy()

    for col in ["UnitPrice", "Quantity"]:
        if col not in df.columns:
            raise KeyError(f"Thiếu cột {col} trong dữ liệu để rời rạc hóa.")

    labels_map = {
        2: ["low", "high"],
        3: ["low", "medium", "high"],
        4: ["low", "medium_low", "medium_high", "high"],
        5: ["very_low", "low", "medium", "high", "very_high"],
    }
    labels = labels_map.get(n_bins, [f"bin_{i}" for i in range(n_bins)])

    for col in ["UnitPrice", "Quantity"]:
        bin_col = f"{col}_bin"
        try:
            df[bin_col] = pd.qcut(df[col], q=n_bins, labels=labels, duplicates="drop")
            logger.info("Đã rời rạc hóa cột %s thành %s với %d bins.", col, bin_col, n_bins)
        except ValueError:
            logger.warning(
                "Không đủ giá trị khác nhau để rời rạc hóa cột %s với %d bins. Bỏ qua.",
                col,
                n_bins,
            )

    return df


__all__ = ["load_clean_data", "build_rfm_features", "compute_product_return_rate", "discretize_features"]

