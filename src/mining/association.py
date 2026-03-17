import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from mlxtend.frequent_patterns import apriori, association_rules


logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def _load_config(config_path: str) -> tuple[dict, Path]:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    project_root = cfg_path.resolve().parent.parent
    logger.info("Đã load cấu hình từ %s", cfg_path)
    return cfg, project_root


def load_clean_data(config_path: str = "configs/params.yaml") -> pd.DataFrame:
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


def _build_return_basket(df: pd.DataFrame) -> pd.DataFrame:
    """Vector hóa giỏ hàng cho các giao dịch trả hàng (is_return=1).

    - Mỗi dòng: một InvoiceNo (hóa đơn có ít nhất một dòng trả hàng).
    - Mỗi cột: một sản phẩm (Description).
    - Giá trị 0/1: sản phẩm đó có xuất hiện trong hóa đơn trả hàng hay không.
    """
    if "is_return" not in df.columns:
        raise KeyError("Thiếu cột is_return trong dữ liệu đã làm sạch.")
    if "InvoiceNo" not in df.columns:
        raise KeyError("Thiếu cột InvoiceNo trong dữ liệu.")
    if "Description" not in df.columns:
        raise KeyError("Thiếu cột Description trong dữ liệu.")

    df_ret = df[df["is_return"] == 1].copy()
    logger.info("Số dòng giao dịch trả hàng dùng cho luật kết hợp: %d", len(df_ret))

    df_ret["item"] = df_ret["Description"].astype(str).str.strip()

    basket = (
        df_ret.groupby(["InvoiceNo", "item"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    basket = (basket > 0).astype(int)

    logger.info(
        "Ma trận giỏ hàng có %d hóa đơn và %d sản phẩm.",
        basket.shape[0],
        basket.shape[1],
    )
    return basket


def _build_invoice_basket_with_target(df: pd.DataFrame, target_item: str = "is_return=1") -> pd.DataFrame:
    """Tạo basket theo hóa đơn và thêm 1 item đại diện cho nhãn trả hàng.

    - Mỗi dòng: một InvoiceNo.
    - Cột: các sản phẩm (Description) + 1 cột `target_item`.
    - `target_item` = 1 nếu hóa đơn có ít nhất 1 dòng is_return=1.
    """
    required = {"InvoiceNo", "Description", "Quantity", "is_return"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Thiếu các cột cần thiết để tạo basket: {sorted(list(missing))}")

    dfx = df.copy()
    dfx["InvoiceNo"] = dfx["InvoiceNo"].astype(str)
    dfx["item"] = dfx["Description"].astype(str).str.strip()

    basket_items = (
        dfx.groupby(["InvoiceNo", "item"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )
    basket_items = (basket_items > 0)

    invoice_target = dfx.groupby("InvoiceNo")["is_return"].max().astype(bool)
    basket_items[target_item] = invoice_target.reindex(basket_items.index).fillna(False)

    logger.info(
        "Basket (invoice) có %d hóa đơn và %d items (gồm target).",
        basket_items.shape[0],
        basket_items.shape[1],
    )
    return basket_items


def mine_rules_consequent_is_return(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
    top_k: int = 10,
    target_item: str = "is_return=1",
    min_support: Optional[float] = None,
    min_confidence: Optional[float] = None,
    min_lift: Optional[float] = None,
    max_len: int = 2,
    max_items: int = 200,
) -> pd.DataFrame:
    """Khai phá luật kết hợp trên toàn bộ hóa đơn, lọc luật có consequent chứa target_item."""
    cfg, _ = _load_config(config_path)
    assoc_cfg = cfg.get("association_rules", {})
    _min_support = float(assoc_cfg.get("min_support", 0.01)) if min_support is None else float(min_support)
    _min_conf = float(assoc_cfg.get("min_confidence", 0.3)) if min_confidence is None else float(min_confidence)
    _min_lift = float(assoc_cfg.get("min_lift", 1.1)) if min_lift is None else float(min_lift)

    if df is None:
        df = load_clean_data(config_path)
    else:
        df = df.copy()

    basket = _build_invoice_basket_with_target(df, target_item=target_item)
    if basket.empty or basket.shape[1] <= 1:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    # Giảm số lượng items để tránh bùng nổ tổ hợp (memory explosion)
    # Giữ lại `max_items` sản phẩm phổ biến nhất + target_item.
    if max_items is not None and max_items > 0 and basket.shape[1] > (max_items + 1):
        item_cols = [c for c in basket.columns if c != target_item]
        item_freq = basket[item_cols].sum(axis=0).sort_values(ascending=False)
        keep_items = item_freq.head(max_items).index.tolist()
        basket = basket[keep_items + [target_item]]
        logger.info("Đã giới hạn basket còn %d sản phẩm phổ biến + target.", len(keep_items))

    # Giới hạn độ dài itemset để kiểm soát độ phức tạp (mặc định: 2 -> luật 1->1, 2->1)
    freq_itemsets = apriori(
        basket.astype(bool),
        min_support=_min_support,
        use_colnames=True,
        max_len=max_len,
    )
    if freq_itemsets.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=_min_conf)
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["consequents"].apply(lambda s: target_item in s)].copy()
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["lift"] >= _min_lift].copy()
    if rules.empty:
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(
        "lift", ascending=False
    )
    if top_k is not None and top_k > 0:
        result = result.head(top_k)
    return result


def mine_return_association_rules(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "configs/params.yaml",
    top_k: int = 10,
) -> pd.DataFrame:
    """Khai phá luật kết hợp cho các giỏ hàng trả hàng (is_return=1) bằng Apriori.

    Trả về bảng gồm: antecedents, consequents, support, confidence, lift.
    """
    cfg, _ = _load_config(config_path)
    assoc_cfg = cfg.get("association_rules", {})
    min_support = float(assoc_cfg.get("min_support", 0.01))
    min_confidence = float(assoc_cfg.get("min_confidence", 0.3))
    min_lift = float(assoc_cfg.get("min_lift", 1.1))

    if df is None:
        df = load_clean_data(config_path)
    else:
        df = df.copy()

    basket = _build_return_basket(df)
    if basket.empty or basket.shape[1] == 0:
        logger.warning("Ma trận giỏ hàng rỗng, không thể khai phá luật kết hợp.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    logger.info(
        "Chạy Apriori với min_support=%.4f để tìm tập mục thường xuyên...",
        min_support,
    )
    freq_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

    if freq_itemsets.empty:
        logger.warning("Không tìm thấy tập mục thường xuyên với min_support đã cho.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    logger.info("Đang sinh luật kết hợp từ tập mục thường xuyên...")
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)

    if rules.empty:
        logger.warning("Không tìm thấy luật thỏa mãn ngưỡng confidence đã cho.")
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        logger.warning("Không có luật nào đạt min_lift=%.3f.", min_lift)
        return pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

    # Chuyển frozenset -> chuỗi cho dễ đọc
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(list(s))))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(list(s))))

    result = rules[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values(
        "lift", ascending=False
    )

    if top_k is not None and top_k > 0:
        result = result.head(top_k)

    logger.info("Đã trích xuất %d luật kết hợp liên quan tới trả hàng.", len(result))
    return result


__all__ = ["load_clean_data", "mine_return_association_rules", "mine_rules_consequent_is_return"]

