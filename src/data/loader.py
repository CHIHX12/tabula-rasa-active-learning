"""
src/data/loader.py
------------------
Data loading, EDA statistics, and stratified K-Fold splitting.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
FEATURES = ["Ni", "Fe", "Co", "Ce"]
TARGET   = "J10"


# ─────────────────────────────────────────────
# Main loading function
# ─────────────────────────────────────────────
def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load OER_database.csv and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Basic cleaning
    df = df.dropna()
    df = df[(df[FEATURES] >= 0).all(axis=1)]
    df = df[df[TARGET] > 0].reset_index(drop=True)

    # Verify composition sum constraint (allow +-0.05 tolerance)
    comp_sum = df[FEATURES].sum(axis=1)
    out_of_range = ((comp_sum < 0.90) | (comp_sum > 1.05)).sum()
    if out_of_range > 0:
        print(f"[Warning] {out_of_range} samples have composition sum outside [0.90, 1.05]")

    print(f"[Data] Loaded {len(df)} samples")
    return df


# ─────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────
def run_eda(df: pd.DataFrame, save_dir: str | Path | None = None) -> None:
    """Generate EDA statistics and plots."""
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("EDA Summary Statistics")
    print("=" * 50)
    print(df.describe().round(3).to_string())

    # J10 distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df[TARGET], bins=40, color="#2196F3", edgecolor="white", alpha=0.85)
    axes[0].set_title("J10 Distribution")
    axes[0].set_xlabel("J10 (mV)")
    axes[0].set_ylabel("Count")

    # J10 vs Ce scatter
    sc = axes[1].scatter(df["Ce"], df[TARGET], c=df["Ni"],
                         cmap="viridis", alpha=0.5, s=10)
    plt.colorbar(sc, ax=axes[1], label="Ni content")
    axes[1].set_title("J10 vs Ce (colored by Ni)")
    axes[1].set_xlabel("Ce")
    axes[1].set_ylabel("J10 (mV)")

    plt.tight_layout()
    _save_or_show(fig, save_dir, "eda_j10_distribution.png")

    # Correlation matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[FEATURES + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, square=True)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "eda_correlation.png")

    # Pairplot (features vs J10)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, feat in enumerate(FEATURES):
        axes[i].scatter(df[feat], df[TARGET], alpha=0.3, s=8, color="#E91E63")
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("J10 (mV)")
        axes[i].set_title(f"J10 vs {feat}")
    plt.tight_layout()
    _save_or_show(fig, save_dir, "eda_feature_vs_j10.png")

    print("[EDA] Complete\n")


def _save_or_show(fig: plt.Figure, save_dir: Path | None, filename: str) -> None:
    if save_dir:
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
        print(f"  -> Saved: {save_dir / filename}")
    else:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────
# Data splitting
# ─────────────────────────────────────────────
def stratified_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Tuple]]:
    """
    Stratified split by Ce content.

    Returns:
        train_val_df : training/validation set for K-Fold
        test_df      : final hold-out test set
        folds        : [(train_idx, val_idx), ...] for n_folds folds
    """
    # Discretize Ce content into 5 bins for stratification
    ce_bins = pd.cut(df["Ce"], bins=5, labels=False)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=ce_bins,
    )
    train_val_df = train_val_df.reset_index(drop=True)
    test_df      = test_df.reset_index(drop=True)

    # K-Fold
    ce_bins_tv = pd.cut(train_val_df["Ce"], bins=5, labels=False)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    folds = list(skf.split(train_val_df, ce_bins_tv))

    print(f"[Split] Train+Val: {len(train_val_df)} | Test: {len(test_df)} | Folds: {n_folds}")
    return train_val_df, test_df, folds


# ─────────────────────────────────────────────
# Scaler utilities
# ─────────────────────────────────────────────
def fit_target_scaler(y: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(y.reshape(-1, 1))
    return scaler


def scale_target(y: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(y.reshape(-1, 1)).ravel()


def inverse_scale_target(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "OER_database.csv"
    df = load_data(data_path)
    run_eda(df, save_dir="figures/eda")
    train_val, test, folds = stratified_split(df)
    print(f"Fold 0 -> train: {len(folds[0][0])}, val: {len(folds[0][1])}")
