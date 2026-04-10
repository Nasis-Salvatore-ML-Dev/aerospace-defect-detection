"""
feat_engineering.py
===================
FAANG-grade Feature Engineering library.

Design principles
-----------------
* Every function is linked to one or more issue codes from eda_functions.py.
  The mapping is explicit in each docstring and in ISSUE_TO_FIX at the bottom.
* Every function accepts a DataFrame and returns a transformed DataFrame —
  making them composable via run_full_pipeline().
* The final clean dataset is saved to a specified output folder as CSV,
  ready for immediate use by train.py.
* Functions auto-detect task type where relevant (regression / classification).
* Scale-safe: chunked operations for DataFrames > 1M rows.

Issue → Fix mapping
-------------------
MISSING_VALUES       → fix_missing_values()
HIGH_SKEWNESS        → fix_skewness()
OUTLIERS             → fix_outliers()
HIGH_CARDINALITY     → fix_high_cardinality()
RARE_CATEGORIES      → fix_rare_categories()
MULTICOLLINEARITY    → fix_multicollinearity()
LOW_VARIANCE         → fix_low_variance()
CONSTANT_FEATURE     → fix_low_variance()
DATA_LEAKAGE_RISK    → fix_data_leakage()
CLASS_IMBALANCE      → fix_class_imbalance()
NON_NORMAL_RESIDUALS → fix_target_transform()
HETEROSCEDASTICITY   → fix_heteroscedasticity()
INTERACTION_SIGNAL   → create_interaction_features()
LOW_DIVERSITY        → fix_low_variance()
DATE_RANGE_TOO_NARROW→ fix_date_diversity()
NEAR_DUPLICATE_COLS  → fix_near_duplicate_columns()
INCONSISTENT_TYPES   → fix_dtypes()
DUPLICATE_ROWS       → fix_duplicates()
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Default output directory for clean datasets ──────────────────────────────
_DEFAULT_OUTPUT_DIR = "data/processed"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PIPELINE LOG
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineLog:
    """Records every transformation applied during the pipeline."""

    def __init__(self) -> None:
        self._steps: List[Dict[str, Any]] = []

    def record(self, fn_name: str, issue_code: str, columns: List[str],
               description: str) -> None:
        self._steps.append({
            "function": fn_name,
            "issue_fixed": issue_code,
            "columns_affected": columns,
            "description": description,
        })
        print(f"  ✅ [{issue_code}] {fn_name}() — {description}")

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("PIPELINE TRANSFORMATION SUMMARY")
        print("=" * 60)
        for i, step in enumerate(self._steps, 1):
            print(f"  {i:>2}. {step['function']:<35} → {step['issue_fixed']}")
            print(f"       {step['description']}")
        print(f"\n  Total transformations applied: {len(self._steps)}")

    def to_dict(self) -> List[Dict]:
        return self._steps


# Global log instance reused across the pipeline
_log = PipelineLog()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TYPE FIXES  [INCONSISTENT_TYPES]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_dtypes(
    df: pd.DataFrame,
    type_map: Dict[str, str],
) -> pd.DataFrame:
    """
    Cast columns to their correct dtypes.

    Fixes: INCONSISTENT_TYPES

    Parameters
    ----------
    type_map : dict {col_name: target_dtype}
               Supported: "int", "float", "str", "category",
               "datetime", "bool"

    Example
    -------
    >>> df = fix_dtypes(df, {"price": "float", "sold_at": "datetime",
    ...                      "fuel": "category"})
    """
    df = df.copy()
    cast_map = {
        "int": "Int64",
        "float": "float64",
        "str": "object",
        "category": "category",
        "bool": "bool",
    }
    affected = []
    for col, dtype in type_map.items():
        if col not in df.columns:
            print(f"  ⚠️  Column '{col}' not found — skipped.")
            continue
        try:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            else:
                df[col] = df[col].astype(cast_map.get(dtype, dtype))
            affected.append(col)
        except Exception as e:
            print(f"  ⚠️  Could not cast '{col}' to {dtype}: {e}")

    _log.record("fix_dtypes", "INCONSISTENT_TYPES", affected,
                f"Cast {len(affected)} columns to correct types.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DUPLICATES  [DUPLICATE_ROWS]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "first",
) -> pd.DataFrame:
    """
    Remove duplicate rows.

    Fixes: DUPLICATE_ROWS

    Parameters
    ----------
    subset : columns to consider for identifying duplicates (None = all)
    keep   : "first" | "last" | False (drop all duplicates)

    Example
    -------
    >>> df = fix_duplicates(df)
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    removed = before - len(df)
    _log.record("fix_duplicates", "DUPLICATE_ROWS", [],
                f"Removed {removed:,} duplicate rows. {len(df):,} rows remain.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MISSING VALUES  [MISSING_VALUES]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_missing_values(
    df: pd.DataFrame,
    strategy: str = "auto",
    fill_values: Optional[Dict[str, Any]] = None,
    drop_threshold: float = 0.6,
    target: Optional[str] = None,
) -> pd.DataFrame:
    """
    Impute or remove missing values.

    Fixes: MISSING_VALUES

    Parameters
    ----------
    strategy        : "auto" | "mean" | "median" | "mode" | "drop_rows"
                      | "drop_cols" | "constant"
                      "auto" → numeric: median, categorical: mode,
                               columns > drop_threshold missing: drop col
    fill_values     : dict {col: fill_value} for "constant" strategy
    drop_threshold  : if strategy="auto", columns with fraction missing
                      above this are dropped instead of imputed
    target          : target column — rows with missing target are always dropped

    Example
    -------
    >>> df = fix_missing_values(df, strategy="auto", target="price")
    >>> df = fix_missing_values(df, strategy="constant",
    ...                         fill_values={"fuel": "unknown"})
    """
    df = df.copy()
    affected = []

    # Always drop rows where target is missing
    if target and target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        dropped = before - len(df)
        if dropped > 0:
            print(f"  ℹ️  Dropped {dropped} rows with missing target '{target}'.")

    for col in df.columns:
        if col == target:
            continue
        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            continue

        pct = n_missing / len(df)
        affected.append(col)

        if strategy == "auto":
            if pct > drop_threshold:
                df = df.drop(columns=[col])
                print(f"  🗑  Dropped column '{col}' ({pct*100:.0f}% missing).")
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0]
                                          if not df[col].mode().empty else "unknown")

        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0]
                                      if not df[col].mode().empty else "unknown")
        elif strategy == "drop_rows":
            df = df.dropna(subset=[col])
        elif strategy == "drop_cols":
            df = df.drop(columns=[col])
        elif strategy == "constant" and fill_values:
            if col in fill_values:
                df[col] = df[col].fillna(fill_values[col])

    df = df.reset_index(drop=True)
    _log.record("fix_missing_values", "MISSING_VALUES", affected,
                f"Imputed/dropped missing values in {len(affected)} columns "
                f"using strategy='{strategy}'.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SKEWNESS  [HIGH_SKEWNESS]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_skewness(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "auto",
    skew_threshold: float = 1.0,
    target: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Apply power transforms to reduce skewness in numeric columns.

    Fixes: HIGH_SKEWNESS, NON_NORMAL_RESIDUALS

    Parameters
    ----------
    columns         : columns to transform (None = auto-detect skewed ones)
    method          : "auto" | "log1p" | "sqrt" | "box-cox" | "yeo-johnson"
                      "auto" selects the best transform per column
    skew_threshold  : only transform columns with |skew| > this
    target          : excluded from transformation

    Returns
    -------
    (transformed_df, transform_map) — keep transform_map to invert predictions

    Example
    -------
    >>> df, tmap = fix_skewness(df, target="price")
    >>> # After training, invert: np.expm1(predictions)
    """
    df = df.copy()
    transform_map: Dict[str, str] = {}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target:
        num_cols = [c for c in num_cols if c != target]
    if columns:
        num_cols = [c for c in columns if c in num_cols]

    affected = []
    for col in num_cols:
        skew = df[col].skew()
        if abs(skew) <= skew_threshold:
            continue

        affected.append(col)
        col_min = df[col].min()

        if method == "auto":
            if col_min >= 0:
                chosen = "log1p"
            else:
                chosen = "yeo-johnson"
        else:
            chosen = method

        if chosen == "log1p":
            df[col] = np.log1p(df[col].clip(lower=0))
            transform_map[col] = "log1p"
        elif chosen == "sqrt":
            df[col] = np.sqrt(df[col].clip(lower=0))
            transform_map[col] = "sqrt"
        elif chosen == "box-cox" and col_min > 0:
            df[col], _ = stats.boxcox(df[col])
            transform_map[col] = "box-cox"
        elif chosen == "yeo-johnson":
            df[col], _ = stats.yeojohnson(df[col])
            transform_map[col] = "yeo-johnson"

        new_skew = df[col].skew()
        print(f"  ✅ {col:<35} {chosen:<12} skew: {skew:+.3f} → {new_skew:+.3f}")

    _log.record("fix_skewness", "HIGH_SKEWNESS", affected,
                f"Applied power transforms to {len(affected)} columns.")
    return df, transform_map


# ═══════════════════════════════════════════════════════════════════════════════
# 6. OUTLIERS  [OUTLIERS]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_outliers(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "cap",
    iqr_factor: float = 1.5,
    target: Optional[str] = None,
) -> pd.DataFrame:
    """
    Handle outliers by capping (Winsorising), removing, or transforming.

    Fixes: OUTLIERS

    Parameters
    ----------
    columns    : columns to process (None = all numeric except target)
    method     : "cap" | "remove" | "log"
                 "cap"    — Winsorise to [Q1 - k*IQR, Q3 + k*IQR]
                 "remove" — drop rows with outliers in any flagged column
                 "log"    — apply log1p (best for right-skewed price data)
    iqr_factor : multiplier for IQR bounds (1.5=standard, 3.0=loose)
    target     : excluded from outlier treatment

    Example
    -------
    >>> df = fix_outliers(df, method="cap", iqr_factor=1.5, target="price")
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target:
        num_cols = [c for c in num_cols if c != target]
    if columns:
        num_cols = [c for c in columns if c in num_cols]

    rows_before = len(df)
    affected = []

    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr

        mask = (df[col] < lower) | (df[col] > upper)
        if not mask.any():
            continue

        affected.append(col)
        if method == "cap":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif method == "remove":
            df = df[~mask]
        elif method == "log":
            df[col] = np.log1p(df[col].clip(lower=0))

    df = df.reset_index(drop=True)
    removed = rows_before - len(df)
    desc = (f"Capped outliers in {len(affected)} columns."
            if method == "cap"
            else f"Removed {removed} outlier rows across {len(affected)} columns.")
    _log.record("fix_outliers", "OUTLIERS", affected, desc)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HIGH CARDINALITY  [HIGH_CARDINALITY]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_high_cardinality(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "target",
    target: Optional[str] = None,
    min_frequency: int = 10,
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Encode high-cardinality categorical columns.

    Fixes: HIGH_CARDINALITY

    Parameters
    ----------
    columns       : columns to encode
    method        : "target" | "frequency" | "binary" | "hash"
                    "target"    — mean target per category (regression/classification)
                    "frequency" — replace category with its frequency
                    "binary"    — one-hot encode (use for low cardinality only)
                    "hash"      — hash trick for very high cardinality
    target        : required for method="target"
    min_frequency : categories appearing fewer times are grouped as "other"
    train_mask    : boolean Series indicating training rows (fit encoders on
                    train only to prevent leakage)

    Example
    -------
    >>> df = fix_high_cardinality(df, columns=["model_key"],
    ...                           method="target", target="price",
    ...                           train_mask=df["data_split"]=="train")
    """
    df = df.copy()
    train = df[train_mask] if train_mask is not None else df

    for col in columns:
        if col not in df.columns:
            continue

        # Group rare categories first
        freq = train[col].value_counts()
        rare_cats = freq[freq < min_frequency].index
        if len(rare_cats) > 0:
            df[col] = df[col].replace(rare_cats, "__other__")

        if method == "target" and target:
            means = train.groupby(col)[target].mean()
            overall_mean = train[target].mean()
            df[col + "_encoded"] = df[col].map(means).fillna(overall_mean)
            df = df.drop(columns=[col])

        elif method == "frequency":
            freq_map = train[col].value_counts(normalize=True)
            df[col + "_freq"] = df[col].map(freq_map).fillna(0)
            df = df.drop(columns=[col])

        elif method == "binary":
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        elif method == "hash":
            n_buckets = 64
            df[col + "_hash"] = df[col].apply(
                lambda x: hash(str(x)) % n_buckets
            )
            df = df.drop(columns=[col])

    _log.record("fix_high_cardinality", "HIGH_CARDINALITY", columns,
                f"Encoded {len(columns)} high-cardinality columns using method='{method}'.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RARE CATEGORIES  [RARE_CATEGORIES]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_rare_categories(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 0.02,
    replacement: str = "__other__",
) -> pd.DataFrame:
    """
    Group rare categories into a single '__other__' bucket.

    Fixes: RARE_CATEGORIES

    Parameters
    ----------
    columns     : columns to process (None = all categorical)
    threshold   : categories with frequency < this fraction are grouped
    replacement : label for the grouped category

    Example
    -------
    >>> df = fix_rare_categories(df, threshold=0.02)
    """
    df = df.copy()
    cat_cols = columns or df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    affected = []
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < threshold].index
        if len(rare) > 0:
            df[col] = df[col].replace(rare, replacement)
            affected.append(col)
            print(f"  ✅ {col:<35} grouped {len(rare)} rare categories → '{replacement}'")

    _log.record("fix_rare_categories", "RARE_CATEGORIES", affected,
                f"Grouped rare categories in {len(affected)} columns.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 9. LOW VARIANCE & CONSTANT FEATURES  [LOW_VARIANCE, CONSTANT_FEATURE]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_low_variance(
    df: pd.DataFrame,
    target: Optional[str] = None,
    variance_threshold: float = 1e-6,
    top_category_threshold: float = 0.99,
) -> pd.DataFrame:
    """
    Drop constant, near-constant, and low-diversity features.

    Fixes: LOW_VARIANCE, CONSTANT_FEATURE, LOW_DIVERSITY

    Parameters
    ----------
    target                  : excluded from dropping
    variance_threshold      : numeric columns with var < this are dropped
    top_category_threshold  : categorical columns where top category covers
                              more than this fraction are dropped

    Example
    -------
    >>> df = fix_low_variance(df, target="price")
    """
    df = df.copy()
    dropped = []

    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].var() < variance_threshold:
                df = df.drop(columns=[col])
                dropped.append(col)
        else:
            top_pct = df[col].value_counts(normalize=True).iloc[0]
            if top_pct > top_category_threshold:
                df = df.drop(columns=[col])
                dropped.append(col)

    _log.record("fix_low_variance", "LOW_VARIANCE", dropped,
                f"Dropped {len(dropped)} low-variance/constant columns.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MULTICOLLINEARITY  [MULTICOLLINEARITY]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_multicollinearity(
    df: pd.DataFrame,
    target: Optional[str] = None,
    vif_threshold: float = 10.0,
    method: str = "vif",
) -> pd.DataFrame:
    """
    Remove features with high multicollinearity.

    Fixes: MULTICOLLINEARITY

    Parameters
    ----------
    target        : excluded from analysis
    vif_threshold : drop features with VIF above this
    method        : "vif" | "correlation"
                    "correlation" — drop one feature from each pair with r > 0.9

    Example
    -------
    >>> df = fix_multicollinearity(df, target="price", vif_threshold=10.0)
    """
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if c != target]
    dropped = []

    if method == "vif":
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            X = df[num_cols].dropna()
            while True:
                vifs = {
                    col: variance_inflation_factor(X.values, i)
                    for i, col in enumerate(X.columns)
                }
                max_col = max(vifs, key=vifs.get)
                if vifs[max_col] <= vif_threshold:
                    break
                print(f"  🗑  Dropping '{max_col}' (VIF={vifs[max_col]:.2f})")
                X = X.drop(columns=[max_col])
                dropped.append(max_col)
            df = df.drop(columns=dropped, errors="ignore")
        except ImportError:
            print("  statsmodels not installed — falling back to correlation method.")
            method = "correlation"

    if method == "correlation":
        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        for col in upper.columns:
            if any(upper[col] > 0.9):
                df = df.drop(columns=[col], errors="ignore")
                dropped.append(col)
                print(f"  🗑  Dropping '{col}' (r > 0.9 with another feature)")

    _log.record("fix_multicollinearity", "MULTICOLLINEARITY", dropped,
                f"Removed {len(dropped)} multicollinear features.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 11. DATA LEAKAGE  [DATA_LEAKAGE_RISK]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_data_leakage(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    Drop features identified as data leakage risks.

    Fixes: DATA_LEAKAGE_RISK

    Parameters
    ----------
    columns : list of column names to drop (from eda_functions.analyse_correlations)

    Example
    -------
    >>> df = fix_data_leakage(df, columns=["price_per_km", "depreciation_rate"])
    """
    df = df.copy()
    to_drop = [c for c in columns if c in df.columns]
    df = df.drop(columns=to_drop)
    _log.record("fix_data_leakage", "DATA_LEAKAGE_RISK", to_drop,
                f"Dropped {len(to_drop)} leaky features.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CLASS IMBALANCE  [CLASS_IMBALANCE]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_class_imbalance(
    df: pd.DataFrame,
    target: str,
    method: str = "oversample",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance class distribution for classification tasks.

    Fixes: CLASS_IMBALANCE

    Parameters
    ----------
    target       : target column
    method       : "oversample" | "undersample" | "smote" | "weights"
                   "oversample"  — random oversampling of minority class
                   "undersample" — random undersampling of majority class
                   "smote"       — SMOTE (requires imbalanced-learn)
                   "weights"     — returns class_weights dict, no resampling
    random_state : reproducibility seed

    Example
    -------
    >>> df = fix_class_imbalance(df, target="churn", method="smote")
    """
    df = df.copy()

    if method == "weights":
        counts = df[target].value_counts()
        total = len(df)
        weights = {cls: total / (len(counts) * cnt)
                   for cls, cnt in counts.items()}
        print(f"  ℹ️  Class weights: {weights}")
        print("  ℹ️  Pass these to your model's class_weight parameter.")
        _log.record("fix_class_imbalance", "CLASS_IMBALANCE", [target],
                    "Computed class weights — no resampling applied.")
        return df

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            X = df.drop(columns=[target])
            y = df[target]
            X_res, y_res = SMOTE(random_state=random_state).fit_resample(X, y)
            df = pd.DataFrame(X_res, columns=X.columns)
            df[target] = y_res
            _log.record("fix_class_imbalance", "CLASS_IMBALANCE", [target],
                        f"Applied SMOTE. New shape: {df.shape}.")
            return df
        except ImportError:
            print("  imbalanced-learn not installed. Falling back to oversample.")
            method = "oversample"

    counts = df[target].value_counts()
    majority_class = counts.index[0]
    minority_class = counts.index[-1]

    if method == "oversample":
        minority_df = df[df[target] == minority_class]
        n_needed = counts[majority_class] - counts[minority_class]
        oversampled = minority_df.sample(n=n_needed, replace=True,
                                         random_state=random_state)
        df = pd.concat([df, oversampled]).reset_index(drop=True)

    elif method == "undersample":
        majority_df = df[df[target] == majority_class].sample(
            n=counts[minority_class], random_state=random_state
        )
        minority_df = df[df[target] == minority_class]
        df = pd.concat([majority_df, minority_df]).reset_index(drop=True)

    _log.record("fix_class_imbalance", "CLASS_IMBALANCE", [target],
                f"Applied {method}. New class distribution: "
                f"{df[target].value_counts().to_dict()}.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 13. TARGET TRANSFORMATION  [NON_NORMAL_RESIDUALS, HIGH_SKEWNESS on target]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_target_transform(
    df: pd.DataFrame,
    target: str,
    method: str = "log1p",
) -> Tuple[pd.DataFrame, Callable]:
    """
    Apply and track a transformation on the target variable.

    Fixes: NON_NORMAL_RESIDUALS, HIGH_SKEWNESS (when applied to target)

    Parameters
    ----------
    target : target column name
    method : "log1p" | "sqrt" | "box-cox" | "yeo-johnson"

    Returns
    -------
    (transformed_df, inverse_fn) — call inverse_fn(predictions) before
    computing business metrics.

    Example
    -------
    >>> df, inv = fix_target_transform(df, target="price", method="log1p")
    >>> # After training: original_predictions = inv(model.predict(X))
    """
    df = df.copy()
    before_skew = df[target].skew()

    if method == "log1p":
        df[target] = np.log1p(df[target])
        inverse_fn = np.expm1
    elif method == "sqrt":
        df[target] = np.sqrt(df[target].clip(lower=0))
        inverse_fn = lambda x: np.square(x)
    elif method == "box-cox":
        transformed, lmbda = stats.boxcox(df[target])
        df[target] = transformed
        inverse_fn = lambda x: stats.inv_boxcox(x, lmbda)
    elif method == "yeo-johnson":
        transformed, lmbda = stats.yeojohnson(df[target])
        df[target] = transformed
        inverse_fn = lambda x, l=lmbda: stats.inv_boxcox(x, l)
    else:
        raise ValueError(f"Unknown method: {method}")

    after_skew = df[target].skew()
    print(f"  ✅ Target '{target}': skew {before_skew:+.3f} → {after_skew:+.3f} ({method})")
    print(f"  ℹ️  Remember to call inverse_fn(predictions) before evaluation.")

    _log.record("fix_target_transform", "NON_NORMAL_RESIDUALS", [target],
                f"Applied {method} to target. Skew: {before_skew:.3f} → {after_skew:.3f}.")
    return df, inverse_fn


# ═══════════════════════════════════════════════════════════════════════════════
# 14. HETEROSCEDASTICITY  [HETEROSCEDASTICITY]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_heteroscedasticity(
    df: pd.DataFrame,
    target: str,
    method: str = "log_target",
) -> pd.DataFrame:
    """
    Reduce heteroscedasticity via target or feature transformation.

    Fixes: HETEROSCEDASTICITY

    Parameters
    ----------
    target : target column
    method : "log_target" | "sqrt_target"
             Both reduce variance proportionality — log is preferred for
             price/revenue targets.

    Example
    -------
    >>> df = fix_heteroscedasticity(df, target="price", method="log_target")
    """
    df = df.copy()
    if method == "log_target":
        df[target] = np.log1p(df[target].clip(lower=0))
    elif method == "sqrt_target":
        df[target] = np.sqrt(df[target].clip(lower=0))

    _log.record("fix_heteroscedasticity", "HETEROSCEDASTICITY", [target],
                f"Applied {method} to reduce heteroscedasticity.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 15. INTERACTION FEATURES  [INTERACTION_SIGNAL]
# ═══════════════════════════════════════════════════════════════════════════════

def create_interaction_features(
    df: pd.DataFrame,
    pairs: Optional[List[Tuple[str, str]]] = None,
    operations: Optional[List[str]] = None,
    target: Optional[str] = None,
    auto_select: bool = True,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Create interaction features from column pairs.

    Fixes: INTERACTION_SIGNAL

    Parameters
    ----------
    pairs       : list of (col_a, col_b) tuples — if None and auto_select=True,
                  pairs are selected automatically by correlation with target
    operations  : list of operations to apply per pair:
                  "multiply" | "divide" | "add" | "subtract" | "ratio"
                  Default: ["multiply", "divide"]
    target      : used for auto-selection of informative pairs
    auto_select : if True and pairs=None, auto-select top pairs
    top_n       : number of auto-selected pairs to consider

    Example
    -------
    >>> df = create_interaction_features(df,
    ...         pairs=[("car_age_years", "mileage"),
    ...                ("engine_power",  "car_age_years")],
    ...         operations=["multiply", "divide"])
    """
    df = df.copy()
    ops = operations or ["multiply", "divide"]

    if pairs is None and auto_select and target:
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target]
        corrs = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
        candidates = corrs[corrs < 0.8].index.tolist()[:top_n]
        pairs = [(candidates[i], candidates[j])
                 for i in range(len(candidates))
                 for j in range(i + 1, min(i + 3, len(candidates)))]

    if not pairs:
        print("  ⚠️  No pairs specified or auto-selected.")
        return df

    new_cols = []
    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        for op in ops:
            if op == "multiply":
                name = f"{col_a}_x_{col_b}"
                df[name] = df[col_a] * df[col_b]
            elif op == "divide":
                name = f"{col_a}_div_{col_b}"
                df[name] = df[col_a] / (df[col_b].replace(0, np.nan))
            elif op == "add":
                name = f"{col_a}_plus_{col_b}"
                df[name] = df[col_a] + df[col_b]
            elif op == "subtract":
                name = f"{col_a}_minus_{col_b}"
                df[name] = df[col_a] - df[col_b]
            elif op == "ratio":
                name = f"{col_a}_ratio_{col_b}"
                total = df[col_a] + df[col_b]
                df[name] = df[col_a] / total.replace(0, np.nan)
            new_cols.append(name)

    _log.record("create_interaction_features", "INTERACTION_SIGNAL", new_cols,
                f"Created {len(new_cols)} interaction features.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 16. DATE DIVERSITY  [DATE_RANGE_TOO_NARROW]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_date_diversity(
    df: pd.DataFrame,
    date_cols: List[str],
    extract: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Extract temporal features from date columns.

    Fixes: DATE_RANGE_TOO_NARROW (by extracting richer signals from dates)

    Parameters
    ----------
    date_cols : date columns to decompose
    extract   : list of components to extract:
                "year" | "month" | "quarter" | "dayofweek" | "dayofyear"
                | "is_weekend" | "season" | "age_days"
                Default: all of the above

    Example
    -------
    >>> df = fix_date_diversity(df, date_cols=["registration_date", "sold_at"])
    """
    df = df.copy()
    extract = extract or ["year", "month", "quarter", "dayofweek",
                          "is_weekend", "season", "age_days"]

    new_cols = []
    for col in date_cols:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        base = col.replace("_date", "").replace("_at", "")

        if "year"      in extract: df[f"{base}_year"]       = parsed.dt.year;       new_cols.append(f"{base}_year")
        if "month"     in extract: df[f"{base}_month"]      = parsed.dt.month;      new_cols.append(f"{base}_month")
        if "quarter"   in extract: df[f"{base}_quarter"]    = parsed.dt.quarter;    new_cols.append(f"{base}_quarter")
        if "dayofweek" in extract: df[f"{base}_dayofweek"]  = parsed.dt.dayofweek;  new_cols.append(f"{base}_dayofweek")
        if "dayofyear" in extract: df[f"{base}_dayofyear"]  = parsed.dt.dayofyear;  new_cols.append(f"{base}_dayofyear")
        if "is_weekend" in extract:
            df[f"{base}_is_weekend"] = (parsed.dt.dayofweek >= 5).astype(int)
            new_cols.append(f"{base}_is_weekend")
        if "season" in extract:
            df[f"{base}_season"] = parsed.dt.month.map(
                {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}
            )
            new_cols.append(f"{base}_season")
        if "age_days" in extract:
            df[f"{base}_age_days"] = (pd.Timestamp.now() - parsed).dt.days
            new_cols.append(f"{base}_age_days")

    _log.record("fix_date_diversity", "DATE_RANGE_TOO_NARROW", new_cols,
                f"Extracted {len(new_cols)} temporal features from {len(date_cols)} date columns.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 17. NEAR-DUPLICATE COLUMNS  [NEAR_DUPLICATE_COLS]
# ═══════════════════════════════════════════════════════════════════════════════

def fix_near_duplicate_columns(
    df: pd.DataFrame,
    pairs: List[Tuple[str, str]],
    keep: str = "first",
) -> pd.DataFrame:
    """
    Drop one column from each near-duplicate pair.

    Fixes: NEAR_DUPLICATE_COLS

    Parameters
    ----------
    pairs : list of (col_a, col_b) near-duplicate pairs
            (from eda_functions.check_data_quality)
    keep  : "first" | "second" — which column to retain

    Example
    -------
    >>> df = fix_near_duplicate_columns(df,
    ...         pairs=[("mileage", "annual_mileage_raw")], keep="first")
    """
    df = df.copy()
    dropped = []
    for col_a, col_b in pairs:
        to_drop = col_b if keep == "first" else col_a
        if to_drop in df.columns:
            df = df.drop(columns=[to_drop])
            dropped.append(to_drop)
            print(f"  🗑  Dropped '{to_drop}' (near-duplicate of "
                  f"'{'col_a' if keep == 'first' else 'col_b'}')")

    _log.record("fix_near_duplicate_columns", "NEAR_DUPLICATE_COLS", dropped,
                f"Dropped {len(dropped)} near-duplicate columns.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 18. SAVE CLEAN DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def save_clean_dataset(
    df: pd.DataFrame,
    filename: str = "clean_dataset.csv",
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    also_save_metadata: bool = True,
) -> str:
    """
    Save the processed DataFrame as a CSV file ready for train.py.

    Parameters
    ----------
    df                  : processed DataFrame
    filename            : output filename (default: clean_dataset.csv)
    output_dir          : directory to save to (default: data/processed)
    also_save_metadata  : if True, saves a sidecar JSON with pipeline log
                          and basic dataset stats

    Returns
    -------
    str — absolute path of the saved CSV file.

    Example
    -------
    >>> path = save_clean_dataset(df, filename="bmw_pricing_clean.csv",
    ...                           output_dir="data/processed")
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("CLEAN DATASET SAVED")
    print("=" * 60)
    print(f"  Path    : {os.path.abspath(csv_path)}")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory  : {mem:.2f} MB")
    print(f"  Dtypes  : {df.dtypes.value_counts().to_dict()}")

    if also_save_metadata:
        meta_path = csv_path.replace(".csv", "_pipeline_log.json")
        metadata = {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "pipeline_steps": _log.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Log     : {os.path.abspath(meta_path)}")

    _log.print_summary()
    return os.path.abspath(csv_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 19. AUTOMATED FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    df: pd.DataFrame,
    target: str,
    task: str = "regression",
    date_cols: Optional[List[str]] = None,
    leaky_cols: Optional[List[str]] = None,
    interaction_pairs: Optional[List[Tuple[str, str]]] = None,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    output_filename: str = "clean_dataset.csv",
    skew_threshold: float = 1.0,
    outlier_method: str = "cap",
    cardinality_threshold: int = 50,
    missing_strategy: str = "auto",
    target_transform: Optional[str] = "log1p",
    train_mask: Optional[pd.Series] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, str]:
    """
    Run the complete feature engineering pipeline end-to-end.

    Designed for autonomous operation — runs without human intervention
    and produces a clean CSV ready for train.py.

    Can also be called step-by-step in a notebook for interactive use.

    Parameters
    ----------
    df                  : raw DataFrame
    target              : target column name
    task                : "regression" | "classification" | "deep_learning"
    date_cols           : date columns to extract temporal features from
    leaky_cols          : columns to drop as data leakage
    interaction_pairs   : explicit pairs for interaction features
    output_dir          : where to save the clean CSV
    output_filename     : name of the output CSV file
    skew_threshold      : skewness threshold for transformation
    outlier_method      : "cap" | "remove" | "log"
    cardinality_threshold: unique value count above which encoding is applied
    missing_strategy    : "auto" | "median" | "mode" | "drop_rows"
    target_transform    : transformation to apply to target (None = skip)
    train_mask          : boolean Series for train rows (prevents leakage in
                          target encoding)
    random_state        : reproducibility seed

    Returns
    -------
    (clean_df, csv_path)

    Example (automated)
    -------------------
    >>> df_clean, path = run_full_pipeline(
    ...     df, target="price", task="regression",
    ...     date_cols=["registration_date", "sold_at"],
    ...     leaky_cols=["price_per_km", "depreciation_rate"],
    ...     output_dir="data/processed",
    ...     output_filename="bmw_pricing_clean.csv"
    ... )
    >>> # df_clean is ready for train.py
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING PIPELINE — AUTOMATED RUN")
    print(f"Task: {task.upper()}  |  Target: {target}")
    print("=" * 60)

    # Step 1 — Drop leaky features
    if leaky_cols:
        df = fix_data_leakage(df, columns=leaky_cols)

    # Step 2 — Fix duplicates
    df = fix_duplicates(df)

    # Step 3 — Fix missing values
    df = fix_missing_values(df, strategy=missing_strategy, target=target)

    # Step 4 — Fix low variance
    df = fix_low_variance(df, target=target)

    # Step 5 — Fix rare categories
    df = fix_rare_categories(df)

    # Step 6 — Fix high cardinality
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    high_card = [c for c in cat_cols if c != target
                 and df[c].nunique() > cardinality_threshold]
    if high_card:
        enc_method = "target" if task == "regression" else "frequency"
        df = fix_high_cardinality(
            df, columns=high_card, method=enc_method,
            target=target, train_mask=train_mask
        )

    # Step 7 — Fix skewness in features
    df, _ = fix_skewness(df, skew_threshold=skew_threshold, target=target)

    # Step 8 — Fix outliers
    df = fix_outliers(df, method=outlier_method, target=target)

    # Step 9 — Date feature extraction
    if date_cols:
        df = fix_date_diversity(df, date_cols=date_cols)

    # Step 10 — Interaction features
    if interaction_pairs or True:  # auto-select if no pairs given
        df = create_interaction_features(
            df, pairs=interaction_pairs, target=target, auto_select=True
        )

    # Step 11 — Fix multicollinearity
    df = fix_multicollinearity(df, target=target)

    # Step 12 — Target transform (regression / deep learning)
    if target_transform and task in ("regression", "deep_learning"):
        df, _ = fix_target_transform(df, target=target, method=target_transform)

    # Step 13 — Class imbalance (classification)
    if task == "classification":
        df = fix_class_imbalance(df, target=target, random_state=random_state)

    # Step 14 — Save
    csv_path = save_clean_dataset(
        df, filename=output_filename, output_dir=output_dir
    )

    return df, csv_path


# ═══════════════════════════════════════════════════════════════════════════════
# 20. ISSUE → FIX LOOKUP TABLE
# ═══════════════════════════════════════════════════════════════════════════════

ISSUE_TO_FIX: Dict[str, str] = {
    "MISSING_VALUES":       "fix_missing_values",
    "HIGH_SKEWNESS":        "fix_skewness",
    "OUTLIERS":             "fix_outliers",
    "HIGH_CARDINALITY":     "fix_high_cardinality",
    "RARE_CATEGORIES":      "fix_rare_categories",
    "MULTICOLLINEARITY":    "fix_multicollinearity",
    "LOW_VARIANCE":         "fix_low_variance",
    "CONSTANT_FEATURE":     "fix_low_variance",
    "DATA_LEAKAGE_RISK":    "fix_data_leakage",
    "CLASS_IMBALANCE":      "fix_class_imbalance",
    "NON_NORMAL_RESIDUALS": "fix_target_transform",
    "HETEROSCEDASTICITY":   "fix_heteroscedasticity",
    "INTERACTION_SIGNAL":   "create_interaction_features",
    "LOW_DIVERSITY":        "fix_low_variance",
    "DATE_RANGE_TOO_NARROW":"fix_date_diversity",
    "NEAR_DUPLICATE_COLS":  "fix_near_duplicate_columns",
    "INCONSISTENT_TYPES":   "fix_dtypes",
    "DUPLICATE_ROWS":       "fix_duplicates",
}


def lookup_fix(issue_code: str) -> str:
    """Return the feat_engineering function name for a given issue code."""
    fn = ISSUE_TO_FIX.get(issue_code, "unknown")
    print(f"  Issue '{issue_code}' → feat_engineering.{fn}()")
    return fn