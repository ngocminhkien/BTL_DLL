import logging
from dataclasses import dataclass
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


@dataclass
class OutlierConfig:
    quantity_lower: float
    quantity_upper: float
    price_lower: float
    price_upper: float


class DataCleaner:
    """Tiền xử lý dữ liệu Online Retail để phục vụ khai phá & modeling."""

    def __init__(self, config_path: str = "configs/params.yaml") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Xác định project root dựa trên vị trí file cấu hình
        # ví dụ: configs/params.yaml -> project_root là thư mục cha của configs
        self.project_root = self.config_path.resolve().parent.parent

        paths_cfg = self.config.get("paths", {})
        self.raw_data_path = self.project_root / paths_cfg.get("raw_data", "data/raw/data.csv")
        self.cleaned_data_path = self.project_root / paths_cfg.get(
            "cleaned_data", "data/processed/cleaned.csv"
        )

        prep_cfg = self.config.get("preprocessing", {})
        out_cfg = prep_cfg.get("outlier", {})

        self.outlier_cfg = OutlierConfig(
            quantity_lower=float(out_cfg.get("quantity", {}).get("lower_quantile", 0.01)),
            quantity_upper=float(out_cfg.get("quantity", {}).get("upper_quantile", 0.99)),
            price_lower=float(out_cfg.get("unit_price", {}).get("lower_quantile", 0.01)),
            price_upper=float(out_cfg.get("unit_price", {}).get("upper_quantile", 0.99)),
        )

        self.drop_missing_customerid = bool(
            prep_cfg.get("customer", {}).get("drop_missing_customerid", True)
        )

        self.df_raw: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {self.config_path}")
        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        logger.info("Đã load cấu hình từ %s", self.config_path)
        return cfg

    # 1. Load dữ liệu
    def load_data(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.raw_data_path}")

        logger.info("Đang đọc dữ liệu từ %s", self.raw_data_path)
        # Dataset Online Retail dùng định dạng ngày dd/mm/yyyy
        df = pd.read_csv(
            self.raw_data_path,
            parse_dates=["InvoiceDate"],
            dayfirst=True,
            dtype={"InvoiceNo": str, "StockCode": str},
            encoding="ISO-8859-1",
        )

        self.df_raw = df.copy()
        self.df = df.copy()

        logger.info("Đã load %d dòng, %d cột", len(df), df.shape[1])
        return self.df

    # 2. Xử lý missing
    def handle_missing(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dữ liệu chưa được load. Hãy gọi load_data() trước.")

        df = self.df

        # CustomerID
        if "CustomerID" in df.columns:
            n_missing_cust = df["CustomerID"].isna().sum()
            if n_missing_cust > 0:
                if self.drop_missing_customerid:
                    logger.info(
                        "Đang loại bỏ %d dòng thiếu CustomerID theo cấu hình.",
                        n_missing_cust,
                    )
                    df = df[~df["CustomerID"].isna()].copy()
                else:
                    logger.info(
                        "Đang gán nhãn 'Guest_Unknown' cho %d dòng thiếu CustomerID.",
                        n_missing_cust,
                    )
                    # Gán -1 rồi chuyển sang chuỗi, sau đó thay -1 bằng Guest_Unknown
                    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int).astype(str)
                    df.loc[df["CustomerID"] == "-1", "CustomerID"] = "Guest_Unknown"

        else:
            logger.warning("Không tìm thấy cột CustomerID trong dữ liệu.")

        # Description
        if "Description" in df.columns:
            n_missing_desc = df["Description"].isna().sum()
            if n_missing_desc > 0:
                logger.info(
                    "Đang gán 'Unknown' cho %d dòng thiếu Description.",
                    n_missing_desc,
                )
                df["Description"] = df["Description"].fillna("Unknown")

            # Chuẩn hóa mô tả để giảm nhiễu (tùy chọn)
            df["Description"] = df["Description"].astype(str).str.strip()
        else:
            logger.warning("Không tìm thấy cột Description trong dữ liệu.")

        self.df = df
        logger.info("Hoàn tất xử lý missing. Còn lại %d dòng.", len(self.df))
        return self.df

    # 3. Xác định giao dịch trả hàng & chuẩn hóa Quantity/UnitPrice
    def identify_returns(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dữ liệu chưa được load. Hãy gọi load_data() trước.")

        df = self.df

        # Đảm bảo InvoiceNo là chuỗi
        if "InvoiceNo" in df.columns:
            df["InvoiceNo"] = df["InvoiceNo"].astype(str)
        else:
            raise KeyError("Không tìm thấy cột InvoiceNo trong dữ liệu.")

        if "Quantity" not in df.columns or "UnitPrice" not in df.columns:
            raise KeyError("Thiếu cột Quantity hoặc UnitPrice trong dữ liệu.")

        logger.info("Đang tạo cột is_return...")
        is_return_cond = (df["Quantity"] < 0) | df["InvoiceNo"].str.startswith("C", na=False)
        df["is_return"] = is_return_cond.astype(int)

        n_returns = int(df["is_return"].sum())
        logger.info("Phát hiện %d giao dịch trả hàng (is_return=1).", n_returns)

        # Sau khi gán nhãn, chuyển Quantity và UnitPrice sang giá trị tuyệt đối
        df["Quantity"] = df["Quantity"].abs()
        df["UnitPrice"] = df["UnitPrice"].abs()

        self.df = df
        return self.df

    # 4. Xử lý outliers & UnitPrice = 0
    def clean_outliers(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Dữ liệu chưa được load. Hãy gọi load_data() trước.")

        df = self.df

        # Loại bỏ UnitPrice <= 0
        n_zero_price = int((df["UnitPrice"] <= 0).sum())
        if n_zero_price > 0:
            logger.info(
                "Loại bỏ %d dòng có UnitPrice <= 0 (giá bằng 0 hoặc âm).",
                n_zero_price,
            )
            df = df[df["UnitPrice"] > 0].copy()

        # Winsorize Quantity & UnitPrice theo quantile cấu hình
        q_low_qty = df["Quantity"].quantile(self.outlier_cfg.quantity_lower)
        q_high_qty = df["Quantity"].quantile(self.outlier_cfg.quantity_upper)
        q_low_price = df["UnitPrice"].quantile(self.outlier_cfg.price_lower)
        q_high_price = df["UnitPrice"].quantile(self.outlier_cfg.price_upper)

        logger.info(
            "Ngưỡng outlier Quantity: [%.2f, %.2f]; UnitPrice: [%.2f, %.2f]",
            q_low_qty,
            q_high_qty,
            q_low_price,
            q_high_price,
        )

        before_qty_extreme = int(((df["Quantity"] < q_low_qty) | (df["Quantity"] > q_high_qty)).sum())
        before_price_extreme = int(
            ((df["UnitPrice"] < q_low_price) | (df["UnitPrice"] > q_high_price)).sum()
        )

        df["Quantity"] = df["Quantity"].clip(q_low_qty, q_high_qty)
        df["UnitPrice"] = df["UnitPrice"].clip(q_low_price, q_high_price)

        logger.info(
            "Đã winsorize %d giá trị Quantity cực trị và %d giá trị UnitPrice cực trị.",
            before_qty_extreme,
            before_price_extreme,
        )

        self.df = df
        return self.df

    def run_full_cleaning(self, save: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chạy toàn bộ pipeline làm sạch và trả về (df_raw, df_clean)."""
        df_raw = self.load_data()
        self.handle_missing()
        self.identify_returns()
        df_clean = self.clean_outliers()

        if save:
            self.cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
            df_clean.to_csv(self.cleaned_data_path, index=False)
            logger.info("Đã lưu dữ liệu đã làm sạch vào %s", self.cleaned_data_path)

        return df_raw, df_clean


__all__ = ["DataCleaner", "OutlierConfig"]

