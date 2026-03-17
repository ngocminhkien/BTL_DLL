from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import sys
import subprocess
import shutil

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root on sys.path so `import src.*` works when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaner import DataCleaner


@dataclass(frozen=True)
class OutputPaths:
    root: Path
    figures: Path
    models: Path
    metrics: Path
    notebooks: Path


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_outputs(project_root: Path) -> OutputPaths:
    out_root = project_root / "outputs"
    figures = out_root / "figures"
    models = out_root / "models"
    metrics = out_root / "metrics"
    notebooks = out_root / "notebooks"
    figures.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    notebooks.mkdir(parents=True, exist_ok=True)
    return OutputPaths(out_root, figures, models, metrics, notebooks)


def strip_notebook_outputs(notebook_path: Path) -> None:
    """Remove outputs/execution counts to avoid nbformat validation issues."""
    import nbformat

    nb = nbformat.read(notebook_path, as_version=4)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    nbformat.write(nb, notebook_path)


def execute_notebook(project_root: Path, notebook_path: Path, out_dir: Path) -> Path:
    """Execute a notebook and write executed copy to outputs/notebooks/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = notebook_path.name
    out_path = out_dir / out_name

    # Work on a copied notebook to avoid modifying the source notebook in repo
    shutil.copy2(notebook_path, out_path)
    strip_notebook_outputs(out_path)

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(out_path),
    ]
    subprocess.run(cmd, cwd=str(project_root), check=True)
    return out_path


def build_invoice_level_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Tạo bộ features ở mức hóa đơn để modeling (tránh leakage).

    - Target: is_return theo hóa đơn = max(is_return) trên các dòng của InvoiceNo
    - Features: tổng Quantity, tổng Monetary, số item unique, thời gian (Hour/Day/Month), Country
    """
    df = clean_df.copy()
    # Ensure datetime
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["Country"] = df["Country"].astype(str)
    df["Description"] = df["Description"].astype(str)

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["HourOfDay"] = df["InvoiceDate"].dt.hour
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["Month"] = df["InvoiceDate"].dt.month

    agg = df.groupby("InvoiceNo").agg(
        CustomerID=("CustomerID", "first"),
        InvoiceDate=("InvoiceDate", "max"),
        Country=("Country", lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0]),
        n_unique_items=("Description", "nunique"),
        total_qty=("Quantity", "sum"),
        avg_unit_price=("UnitPrice", "mean"),
        monetary=("TotalPrice", "sum"),
        HourOfDay=("HourOfDay", "median"),
        DayOfWeek=("DayOfWeek", "median"),
        Month=("Month", "median"),
        is_return=("is_return", "max"),
    )
    agg = agg.reset_index()
    agg["is_return"] = agg["is_return"].astype(int)
    return agg


def split_xy(features_df: pd.DataFrame, test_size: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    drop_cols = [c for c in ["is_return", "InvoiceNo", "InvoiceDate", "CustomerID"] if c in features_df.columns]
    X = features_df.drop(columns=drop_cols)
    y = features_df["is_return"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, X_test, y_train, y_test


def encode_for_models(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot Country; keep numeric columns as-is."""
    # Ensure consistent dtypes
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train["Country"] = X_train["Country"].astype(str)
    X_test["Country"] = X_test["Country"].astype(str)

    X_train_enc = pd.get_dummies(X_train, columns=["Country"], drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=["Country"], drop_first=True)
    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    return X_train_enc, X_test_enc


def train_models(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> Dict[str, object]:
    models: Dict[str, object] = {}

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=seed)
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    try:
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=seed,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
    except Exception as e:
        print("[WARN] XGBoost training skipped:", e)

    return models


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    out = {
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if proba is not None else None,
        "pr_auc": float(average_precision_score(y_test, proba)) if proba is not None else None,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return out


def plot_confusion_matrix(cm: np.ndarray, save_path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No return", "Return"],
        yticklabels=["No return", "Return"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "params.yaml"
    cfg = load_config(config_path)

    outputs = ensure_outputs(project_root)

    # 1) Clean data
    cleaner = DataCleaner(config_path=str(config_path))
    _, clean_df = cleaner.run_full_cleaning(save=True)

    # 2) Build features
    features_df = build_invoice_level_features(clean_df)

    # Save features (parquet if available, else csv)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    features_out_parquet = outputs.metrics / f"features_{ts}.parquet"
    try:
        features_df.to_parquet(features_out_parquet, index=False)
        features_saved = str(features_out_parquet)
    except Exception:
        features_out_csv = outputs.metrics / f"features_{ts}.csv"
        features_df.to_csv(features_out_csv, index=False)
        features_saved = str(features_out_csv)

    # 3) Split and encode
    test_size = float(cfg.get("modeling", {}).get("test_size", 0.2))
    seed = int(cfg.get("random_seed", 42))
    X_train, X_test, y_train, y_test = split_xy(features_df, test_size=test_size, seed=seed)
    X_train_enc, X_test_enc = encode_for_models(X_train, X_test)

    # Persist test set for evaluation notebook
    X_test_enc.to_csv(outputs.metrics / f"X_test_{ts}.csv", index=False)
    pd.Series(y_test, name="is_return").to_csv(outputs.metrics / f"y_test_{ts}.csv", index=False)

    # 4) Train
    models = train_models(X_train_enc, y_train, seed=seed)

    # 5) Evaluate and save
    metrics = {}
    best_name = None
    best_f1 = -1.0
    for name, model in models.items():
        m = evaluate_model(model, X_test_enc, y_test)
        metrics[name] = m
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_name = name

        # Save model
        joblib.dump(model, outputs.models / f"{name}_{ts}.pkl")

        # Save confusion matrix plot
        cm = np.array(m["confusion_matrix"])
        plot_confusion_matrix(
            cm,
            outputs.figures / f"confusion_matrix_{name}_{ts}.png",
            title=f"Confusion Matrix - {name}",
        )

    summary_rows = [
        {"model": k, "f1": v["f1"], "roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]}
        for k, v in metrics.items()
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("f1", ascending=False)
    summary_df.to_csv(outputs.metrics / f"metrics_summary_{ts}.csv", index=False)

    report_path = outputs.metrics / f"metrics_full_{ts}.json"
    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if best_name is not None:
        joblib.dump(models[best_name], outputs.models / "best_model.pkl")

    run_info = {
        "timestamp": ts,
        "features_saved": features_saved,
        "best_model": best_name,
        "outputs": {
            "root": str(outputs.root),
            "figures": str(outputs.figures),
            "models": str(outputs.models),
            "metrics": str(outputs.metrics),
        },
    }
    (outputs.metrics / f"run_info_{ts}.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[DONE] Pipeline completed.")
    print("Best model:", best_name, "F1:", best_f1)
    print("Outputs saved to:", outputs.root)

    # 6) Execute notebooks 01 -> 05 (and 04b if exists)
    notebooks_to_run = [
        project_root / "notebooks" / "01_eda.ipynb",
        project_root / "notebooks" / "02_preprocess_feature.ipynb",
        project_root / "notebooks" / "03_mining_or_clustering.ipynb",
        project_root / "notebooks" / "04_modeling.ipynb",
        project_root / "notebooks" / "04b_semi_supervised.ipynb",
        project_root / "notebooks" / "05_evaluation.ipynb",
    ]
    existing = [p for p in notebooks_to_run if p.exists()]
    print(f"[NOTEBOOKS] Executing {len(existing)} notebooks -> {outputs.notebooks}")
    for nb in existing:
        try:
            out_nb = execute_notebook(project_root, nb, outputs.notebooks)
            print("[OK] Executed:", nb.name, "->", out_nb)
        except Exception as e:
            print("[WARN] Notebook failed:", nb.name, "error:", e)


if __name__ == "__main__":
    main()

