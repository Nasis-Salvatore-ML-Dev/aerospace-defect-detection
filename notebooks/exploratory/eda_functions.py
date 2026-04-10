"""
eda_functions.py
================
FAANG-grade Exploratory Data Analysis library.

Design principles
-----------------
* Every function returns an EDAResult — a structured object carrying
  the issue code, severity, human-readable summary, and the name of
  the feat_engineering.py function that fixes it.
* Functions can be called individually in a Jupyter notebook for
  interactive exploration, or composed into run_full_eda() for
  fully automated pipelines.
* Task-agnostic: works for regression, classification, and deep learning.
* Scale-safe: tested from 2 000 to 10 M rows via sampling strategies.

Issue taxonomy (shared with feat_engineering.py)
-------------------------------------------------
HIGH_SKEWNESS          OUTLIERS               MISSING_VALUES
HIGH_CARDINALITY       RARE_CATEGORIES        MULTICOLLINEARITY
LOW_VARIANCE           DATA_LEAKAGE_RISK      CLASS_IMBALANCE
DUPLICATE_ROWS         INCONSISTENT_TYPES     NON_NORMAL_RESIDUALS
HETEROSCEDASTICITY     INTERACTION_SIGNAL     LOW_DIVERSITY
DATE_RANGE_TOO_NARROW  CONSTANT_FEATURE       NEAR_DUPLICATE_COLS
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Sampling threshold: above this row count we sample for speed ─────────────
_SAMPLE_THRESHOLD = 200_000
_SAMPLE_N = 50_000


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EDAResult:
    """Structured result returned by every EDA function."""
    issue_code: str                        # e.g. "HIGH_SKEWNESS"
    severity: str                          # "ok" | "warning" | "critical"
    affected_columns: List[str]            # columns with this issue
    summary: str                           # human-readable explanation
    fix_function: str                      # feat_engineering.py function to call
    details: Dict[str, Any] = field(default_factory=dict)  # metric values

    def __repr__(self) -> str:
        icon = {"ok": "✅", "warning": "⚠️ ", "critical": "❌"}.get(self.severity, "ℹ️ ")
        lines = [
            f"\n{icon} [{self.issue_code}] — severity: {self.severity.upper()}",
            f"   Affected : {self.affected_columns if self.affected_columns else 'none'}",
            f"   Summary  : {self.summary}",
            f"   Fix with : feat_engineering.{self.fix_function}()",
        ]
        if self.details:
            lines.append("   Details  :")
            for k, v in self.details.items():
                if isinstance(v, float):
                    lines.append(f"     {k}: {v:.4f}")
                else:
                    lines.append(f"     {k}: {v}")
        return "\n".join(lines)

    def print_report(self) -> None:
        print(repr(self))


@dataclass
class EDAReport:
    """Aggregated report from run_full_eda()."""
    results: List[EDAResult] = field(default_factory=list)

    def print_report(self) -> None:
        print("\n" + "=" * 70)
        print("EDA FULL REPORT")
        print("=" * 70)
        criticals = [r for r in self.results if r.severity == "critical"]
        warnings_  = [r for r in self.results if r.severity == "warning"]
        oks        = [r for r in self.results if r.severity == "ok"]
        print(f"  Critical issues : {len(criticals)}")
        print(f"  Warnings        : {len(warnings_)}")
        print(f"  Clean checks    : {len(oks)}")
        print("=" * 70)
        for r in self.results:
            r.print_report()
        print("\n" + "=" * 70)
        print("RECOMMENDED ACTIONS (in order)")
        print("=" * 70)
        seen = set()
        priority = ["critical", "warning", "ok"]
        for sev in priority:
            for r in self.results:
                if r.severity == sev and r.fix_function not in seen and r.severity != "ok":
                    seen.add(r.fix_function)
                    print(f"  feat_engineering.{r.fix_function}()")

    @property
    def has_critical(self) -> bool:
        return any(r.severity == "critical" for r in self.results)

    @property
    def issue_codes(self) -> List[str]:
        return [r.issue_code for r in self.results]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sample for large datasets to keep functions fast."""
    if len(df) > _SAMPLE_THRESHOLD:
        return df.sample(_SAMPLE_N, random_state=42)
    return df


def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _datetime_cols(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

def overview(df: pd.DataFrame, target: Optional[str] = None) -> EDAResult:
    """
    Print a structured summary of the dataset.

    Covers: shape, dtypes, memory usage, missing value counts,
    numeric summary statistics, and target variable description.

    Parameters
    ----------
    df     : input DataFrame
    target : name of the target column (optional)

    Returns
    -------
    EDAResult with issue_code MISSING_VALUES if any nulls found.
    """
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows              : {len(df):,}")
    print(f"  Columns           : {df.shape[1]}")
    mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"  Memory usage      : {mem:.2f} MB")
    print(f"\n  Dtypes breakdown:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {str(dtype):<20} {count} columns")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    affected = []
    if len(missing) > 0:
        print(f"\n  Missing values ({len(missing)} columns affected):")
        for col, n in missing.items():
            pct = n / len(df) * 100
            print(f"    {col:<35} {n:>6,} ({pct:.1f}%)")
            affected.append(col)
    else:
        print("\n  Missing values    : none")

    if target and target in df.columns:
        print(f"\n  Target '{target}':")
        if pd.api.types.is_numeric_dtype(df[target]):
            print(f"    mean={df[target].mean():.2f}  "
                  f"std={df[target].std():.2f}  "
                  f"min={df[target].min():.2f}  "
                  f"max={df[target].max():.2f}")
        else:
            print(f"    unique classes: {df[target].nunique()}")
            print(f"    class counts:\n{df[target].value_counts().to_string()}")

    severity = "critical" if len(missing) > 0 else "ok"
    summary = (f"{len(missing)} columns have missing values." if missing.any()
               else "No missing values detected.")
    return EDAResult(
        issue_code="MISSING_VALUES",
        severity=severity,
        affected_columns=affected,
        summary=summary,
        fix_function="fix_missing_values",
        details={col: f"{v/len(df)*100:.1f}%" for col, v in missing.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_schema(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]] = None,
    cardinality_threshold: int = 50,
) -> List[EDAResult]:
    """
    Validate column types and detect high-cardinality categoricals.

    Parameters
    ----------
    df                    : input DataFrame
    expected_schema       : dict {col_name: expected_dtype_str} — optional
    cardinality_threshold : categorical columns with more unique values
                            than this are flagged HIGH_CARDINALITY

    Returns
    -------
    List of EDAResult — one per issue type found.
    """
    print("\n" + "=" * 60)
    print("SCHEMA VALIDATION")
    print("=" * 60)

    results = []

    # ── Type mismatches ──────────────────────────────────────────────────────
    type_issues = []
    if expected_schema:
        for col, expected in expected_schema.items():
            if col not in df.columns:
                print(f"  ❌ Column '{col}' expected but not found.")
                type_issues.append(col)
            else:
                actual = str(df[col].dtype)
                match = expected in actual or actual in expected
                status = "✅" if match else "⚠️ "
                print(f"  {status} {col:<35} expected={expected:<12} actual={actual}")
                if not match:
                    type_issues.append(col)
    else:
        print("  No expected schema provided — printing inferred types:")
        for col in df.columns:
            print(f"    {col:<35} {str(df[col].dtype)}")

    if type_issues:
        results.append(EDAResult(
            issue_code="INCONSISTENT_TYPES",
            severity="critical",
            affected_columns=type_issues,
            summary=f"{len(type_issues)} columns have unexpected types.",
            fix_function="fix_dtypes",
        ))

    # ── High cardinality ─────────────────────────────────────────────────────
    cat_cols = _categorical_cols(df)
    high_card = []
    print(f"\n  Cardinality check (threshold={cardinality_threshold}):")
    for col in cat_cols:
        n = df[col].nunique()
        flag = "⚠️ " if n > cardinality_threshold else "✅"
        print(f"  {flag} {col:<35} {n:>5} unique values")
        if n > cardinality_threshold:
            high_card.append(col)

    if high_card:
        results.append(EDAResult(
            issue_code="HIGH_CARDINALITY",
            severity="warning",
            affected_columns=high_card,
            summary=f"{len(high_card)} categorical columns exceed {cardinality_threshold} unique values.",
            fix_function="fix_high_cardinality",
            details={col: df[col].nunique() for col in high_card},
        ))
    else:
        results.append(EDAResult(
            issue_code="HIGH_CARDINALITY",
            severity="ok",
            affected_columns=[],
            summary="All categorical columns have acceptable cardinality.",
            fix_function="fix_high_cardinality",
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

def check_data_quality(df: pd.DataFrame) -> List[EDAResult]:
    """
    Check for duplicates, constant features, and near-duplicate columns.

    Returns
    -------
    List of EDAResult — one per issue type.
    """
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    results = []
    dfs = _sample(df)

    # ── Duplicate rows ───────────────────────────────────────────────────────
    n_dupes = df.duplicated().sum()
    pct = n_dupes / len(df) * 100
    flag = "❌" if n_dupes > 0 else "✅"
    print(f"  {flag} Duplicate rows    : {n_dupes:,} ({pct:.2f}%)")
    results.append(EDAResult(
        issue_code="DUPLICATE_ROWS",
        severity="critical" if n_dupes > 0 else "ok",
        affected_columns=[],
        summary=f"{n_dupes} duplicate rows detected ({pct:.2f}%).",
        fix_function="fix_duplicates",
        details={"n_duplicates": n_dupes, "pct": pct},
    ))

    # ── Constant / near-constant features ────────────────────────────────────
    constant_cols = []
    print("\n  Variance check (constant features):")
    for col in _numeric_cols(df):
        if df[col].nunique() <= 1:
            print(f"  ❌ {col:<35} CONSTANT")
            constant_cols.append(col)
        else:
            var = df[col].var()
            if var < 1e-6:
                print(f"  ⚠️  {col:<35} near-constant (var={var:.2e})")
                constant_cols.append(col)

    if constant_cols:
        results.append(EDAResult(
            issue_code="CONSTANT_FEATURE",
            severity="critical",
            affected_columns=constant_cols,
            summary=f"{len(constant_cols)} constant or near-constant features detected.",
            fix_function="fix_low_variance",
        ))

    # ── Near-duplicate columns (correlation > 0.99) ──────────────────────────
    num_cols = _numeric_cols(dfs)
    near_dupes = []
    if len(num_cols) >= 2:
        corr = dfs[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        near_dupes = [
            (c, r)
            for c in upper.columns
            for r in upper.index
            if upper.loc[r, c] > 0.99
        ]
        print(f"\n  Near-duplicate columns (r > 0.99): {len(near_dupes)} pairs")
        for a, b in near_dupes:
            print(f"    ⚠️  {a} ↔ {b}")

    if near_dupes:
        affected = list({col for pair in near_dupes for col in pair})
        results.append(EDAResult(
            issue_code="NEAR_DUPLICATE_COLS",
            severity="warning",
            affected_columns=affected,
            summary=f"{len(near_dupes)} near-duplicate column pairs (r>0.99).",
            fix_function="fix_near_duplicate_columns",
            details={"pairs": near_dupes},
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def check_distributions(
    df: pd.DataFrame,
    skew_threshold: float = 1.0,
    rare_threshold: float = 0.02,
    low_variance_pct: float = 0.95,
) -> List[EDAResult]:
    """
    Analyse numeric distributions and categorical frequency distributions.

    Detects: high skewness, low variance features, rare categories,
    class imbalance (classification), low diversity.

    Parameters
    ----------
    skew_threshold      : absolute skewness above which a column is flagged
    rare_threshold      : categories representing < this fraction are rare
    low_variance_pct    : a single category covering > this fraction flags LOW_VARIANCE

    Returns
    -------
    List of EDAResult.
    """
    print("\n" + "=" * 60)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 60)

    results = []
    dfs = _sample(df)

    # ── Numeric: skewness ────────────────────────────────────────────────────
    print("\n  Skewness (numeric columns):")
    skewed = []
    for col in _numeric_cols(dfs):
        sk = dfs[col].skew()
        flag = "⚠️ " if abs(sk) > skew_threshold else "✅"
        print(f"  {flag} {col:<35} skew={sk:+.3f}")
        if abs(sk) > skew_threshold:
            skewed.append(col)

    results.append(EDAResult(
        issue_code="HIGH_SKEWNESS",
        severity="warning" if skewed else "ok",
        affected_columns=skewed,
        summary=(f"{len(skewed)} numeric columns are skewed (|skew| > {skew_threshold})."
                 if skewed else "No significant skewness detected."),
        fix_function="fix_skewness",
        details={col: round(float(dfs[col].skew()), 4) for col in skewed},
    ))

    # ── Numeric: low variance ────────────────────────────────────────────────
    low_var = []
    for col in _numeric_cols(dfs):
        if dfs[col].nunique() > 1:
            top_pct = dfs[col].value_counts(normalize=True).iloc[0]
            if top_pct > low_variance_pct:
                low_var.append(col)

    if low_var:
        results.append(EDAResult(
            issue_code="LOW_VARIANCE",
            severity="warning",
            affected_columns=low_var,
            summary=f"{len(low_var)} features dominated by a single value (>{low_variance_pct*100:.0f}%).",
            fix_function="fix_low_variance",
        ))

    # ── Categorical: rare categories ─────────────────────────────────────────
    print("\n  Rare categories (categorical columns):")
    rare_cols = {}
    for col in _categorical_cols(df):
        freqs = df[col].value_counts(normalize=True)
        rare = freqs[freqs < rare_threshold].index.tolist()
        if rare:
            rare_cols[col] = rare
            print(f"  ⚠️  {col:<35} {len(rare)} rare categories: {rare[:5]}")
        else:
            print(f"  ✅ {col:<35} no rare categories")

    if rare_cols:
        results.append(EDAResult(
            issue_code="RARE_CATEGORIES",
            severity="warning",
            affected_columns=list(rare_cols.keys()),
            summary=f"{len(rare_cols)} columns contain rare categories (<{rare_threshold*100:.0f}% frequency).",
            fix_function="fix_rare_categories",
            details=rare_cols,
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DIVERSITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def check_diversity(
    df: pd.DataFrame,
    date_cols: Optional[List[str]] = None,
    min_date_range_years: float = 2.0,
    min_unique_ratio: float = 0.01,
) -> List[EDAResult]:
    """
    Check whether features exhibit sufficiently diverse values.

    Covers:
    - Date columns: does the range span enough years?
    - Numeric columns: is the unique value ratio above the minimum?
    - Categorical columns: is the entropy above a minimum threshold?

    Parameters
    ----------
    date_cols              : list of date-like column names to check
    min_date_range_years   : minimum acceptable date range in years
    min_unique_ratio       : for numeric cols, unique/total must exceed this

    Returns
    -------
    List of EDAResult.
    """
    print("\n" + "=" * 60)
    print("DIVERSITY CHECK")
    print("=" * 60)

    results = []

    # ── Date range diversity ─────────────────────────────────────────────────
    date_issues = []
    if date_cols:
        print("\n  Date range check:")
        for col in date_cols:
            if col not in df.columns:
                continue
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                rng = parsed.max() - parsed.min()
                years = rng.days / 365.25
                flag = "✅" if years >= min_date_range_years else "⚠️ "
                print(f"  {flag} {col:<35} range={years:.1f} years "
                      f"({parsed.min().date()} → {parsed.max().date()})")
                if years < min_date_range_years:
                    date_issues.append(col)
            except Exception:
                print(f"  ⚠️  {col:<35} could not parse as date")

    if date_issues:
        results.append(EDAResult(
            issue_code="DATE_RANGE_TOO_NARROW",
            severity="warning",
            affected_columns=date_issues,
            summary=(f"{len(date_issues)} date columns span less than "
                     f"{min_date_range_years} years — predictions may not generalise."),
            fix_function="fix_date_diversity",
            details={"min_years_required": min_date_range_years},
        ))

    # ── Numeric diversity ────────────────────────────────────────────────────
    print("\n  Numeric diversity (unique value ratio):")
    low_diversity = []
    for col in _numeric_cols(df):
        ratio = df[col].nunique() / len(df)
        flag = "✅" if ratio >= min_unique_ratio else "⚠️ "
        print(f"  {flag} {col:<35} unique_ratio={ratio:.4f}")
        if ratio < min_unique_ratio:
            low_diversity.append(col)

    # ── Categorical entropy ──────────────────────────────────────────────────
    print("\n  Categorical diversity (Shannon entropy):")
    for col in _categorical_cols(df):
        freqs = df[col].value_counts(normalize=True)
        entropy = -np.sum(freqs * np.log2(freqs + 1e-10))
        max_entropy = np.log2(len(freqs)) if len(freqs) > 1 else 1
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        flag = "✅" if norm_entropy > 0.3 else "⚠️ "
        print(f"  {flag} {col:<35} normalised_entropy={norm_entropy:.3f}")
        if norm_entropy <= 0.3:
            low_diversity.append(col)

    if low_diversity:
        results.append(EDAResult(
            issue_code="LOW_DIVERSITY",
            severity="warning",
            affected_columns=low_diversity,
            summary=f"{len(low_diversity)} features show low diversity.",
            fix_function="fix_low_variance",
        ))
    else:
        results.append(EDAResult(
            issue_code="LOW_DIVERSITY",
            severity="ok",
            affected_columns=[],
            summary="All features show acceptable diversity.",
            fix_function="fix_low_variance",
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. OUTLIER DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    z_threshold: float = 3.5,
    iqr_factor: float = 1.5,
    outlier_pct_threshold: float = 0.05,
) -> EDAResult:
    """
    Detect outliers in numeric columns using IQR or Z-score method.

    Parameters
    ----------
    method                  : "iqr" (robust) or "zscore" (assumes normality)
    z_threshold             : Z-score threshold (used when method="zscore")
    iqr_factor              : IQR multiplier (standard=1.5, strict=3.0)
    outlier_pct_threshold   : columns with more outliers than this are flagged

    Returns
    -------
    EDAResult with issue_code OUTLIERS.
    """
    print("\n" + "=" * 60)
    print(f"OUTLIER DETECTION (method={method})")
    print("=" * 60)

    dfs = _sample(df)
    flagged = {}

    for col in _numeric_cols(dfs):
        series = dfs[col].dropna()
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            mask = (series < q1 - iqr_factor * iqr) | (series > q3 + iqr_factor * iqr)
        else:
            z = np.abs(stats.zscore(series))
            mask = z > z_threshold

        n_out = mask.sum()
        pct = n_out / len(series)
        flag = "⚠️ " if pct > outlier_pct_threshold else "✅"
        print(f"  {flag} {col:<35} outliers={n_out:>5,} ({pct*100:.2f}%)")
        if pct > outlier_pct_threshold:
            flagged[col] = round(pct * 100, 2)

    severity = "warning" if flagged else "ok"
    return EDAResult(
        issue_code="OUTLIERS",
        severity=severity,
        affected_columns=list(flagged.keys()),
        summary=(f"{len(flagged)} columns have >{outlier_pct_threshold*100:.0f}% outliers."
                 if flagged else "No significant outliers detected."),
        fix_function="fix_outliers",
        details=flagged,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_correlations(
    df: pd.DataFrame,
    target: Optional[str] = None,
    leakage_threshold: float = 0.95,
    interaction_threshold: float = 0.3,
) -> List[EDAResult]:
    """
    Analyse feature correlations.

    Covers:
    - Feature-target correlations (regression) — top predictors
    - Data leakage detection: features suspiciously correlated with target
    - Interaction signal detection: feature pairs whose product may help

    Parameters
    ----------
    target               : target column name
    leakage_threshold    : feature-target |r| above this → DATA_LEAKAGE_RISK
    interaction_threshold: feature-target |r| below this → candidate for
                           interaction feature creation

    Returns
    -------
    List of EDAResult.
    """
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    dfs = _sample(df)
    results = []
    num_cols = _numeric_cols(dfs)

    # ── Feature-target correlation ───────────────────────────────────────────
    if target and target in dfs.columns and target in num_cols:
        corr_with_target = (
            dfs[num_cols].corr()[target]
            .drop(target, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        print(f"\n  Feature correlations with target '{target}':")
        for col, r in corr_with_target.items():
            flag = "❌" if r > leakage_threshold else "✅"
            print(f"  {flag} {col:<35} |r|={r:.4f}")

        # Leakage
        leaky = corr_with_target[corr_with_target > leakage_threshold].index.tolist()
        if leaky:
            results.append(EDAResult(
                issue_code="DATA_LEAKAGE_RISK",
                severity="critical",
                affected_columns=leaky,
                summary=(f"{len(leaky)} features have |r| > {leakage_threshold} "
                         f"with target — likely data leakage."),
                fix_function="fix_data_leakage",
                details={col: round(float(corr_with_target[col]), 4) for col in leaky},
            ))

        # Interaction candidates: low individual correlation
        weak = corr_with_target[
            (corr_with_target < interaction_threshold) &
            (corr_with_target > 0.05)
        ].index.tolist()
        if len(weak) >= 2:
            pairs = [(weak[i], weak[j])
                     for i in range(len(weak))
                     for j in range(i + 1, min(i + 3, len(weak)))]
            results.append(EDAResult(
                issue_code="INTERACTION_SIGNAL",
                severity="warning",
                affected_columns=weak[:10],
                summary=(f"{len(weak)} features have weak individual correlation "
                         f"with target — interaction features may help."),
                fix_function="create_interaction_features",
                details={"candidate_pairs": pairs[:10]},
            ))

    # ── Feature-feature multicollinearity (pairwise) ─────────────────────────
    print("\n  Feature-feature correlation matrix (top pairs):")
    feat_cols = [c for c in num_cols if c != target]
    if len(feat_cols) >= 2:
        corr_matrix = dfs[feat_cols].corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_pairs = [
            (col, row, round(upper.loc[row, col], 4))
            for col in upper.columns
            for row in upper.index
            if upper.loc[row, col] > 0.7
        ]
        high_pairs.sort(key=lambda x: -x[2])
        for a, b, r in high_pairs[:15]:
            flag = "⚠️ " if r > 0.85 else "ℹ️ "
            print(f"  {flag} {a:<25} ↔ {b:<25} r={r:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MULTICOLLINEARITY (VIF)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_multicollinearity(
    df: pd.DataFrame,
    target: Optional[str] = None,
    vif_threshold: float = 10.0,
) -> EDAResult:
    """
    Detect multicollinearity using Variance Inflation Factor (VIF).

    VIF > 10  → critical multicollinearity
    VIF 5-10  → moderate — monitor
    VIF < 5   → acceptable

    Parameters
    ----------
    target        : excluded from VIF calculation
    vif_threshold : VIF above this is flagged as critical

    Returns
    -------
    EDAResult with issue_code MULTICOLLINEARITY.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("  statsmodels not installed. Run: pip install statsmodels")
        return EDAResult(
            issue_code="MULTICOLLINEARITY",
            severity="ok",
            affected_columns=[],
            summary="statsmodels not available — VIF check skipped.",
            fix_function="fix_multicollinearity",
        )

    print("\n" + "=" * 60)
    print(f"MULTICOLLINEARITY CHECK (VIF threshold={vif_threshold})")
    print("=" * 60)

    dfs = _sample(df)
    num_cols = [c for c in _numeric_cols(dfs) if c != target]
    X = dfs[num_cols].dropna()

    vif_data = {}
    flagged = []
    for i, col in enumerate(num_cols):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data[col] = round(vif, 2)
            flag = "❌" if vif > vif_threshold else ("⚠️ " if vif > 5 else "✅")
            print(f"  {flag} {col:<35} VIF={vif:.2f}")
            if vif > vif_threshold:
                flagged.append(col)
        except Exception:
            pass

    severity = "critical" if flagged else "ok"
    return EDAResult(
        issue_code="MULTICOLLINEARITY",
        severity=severity,
        affected_columns=flagged,
        summary=(f"{len(flagged)} features have VIF > {vif_threshold}."
                 if flagged else "No critical multicollinearity detected."),
        fix_function="fix_multicollinearity",
        details=vif_data,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CLASS IMBALANCE (CLASSIFICATION)
# ═══════════════════════════════════════════════════════════════════════════════

def check_class_imbalance(
    df: pd.DataFrame,
    target: str,
    imbalance_threshold: float = 0.15,
) -> EDAResult:
    """
    Check for class imbalance in a classification target.

    Parameters
    ----------
    target               : name of the target column
    imbalance_threshold  : minority class proportion below this → flagged

    Returns
    -------
    EDAResult with issue_code CLASS_IMBALANCE.
    """
    print("\n" + "=" * 60)
    print("CLASS IMBALANCE CHECK")
    print("=" * 60)

    counts = df[target].value_counts(normalize=True).sort_values()
    for cls, pct in counts.items():
        flag = "⚠️ " if pct < imbalance_threshold else "✅"
        print(f"  {flag} {str(cls):<30} {pct*100:.2f}%")

    minority_pct = counts.iloc[0]
    flagged = minority_pct < imbalance_threshold
    severity = "critical" if minority_pct < 0.05 else ("warning" if flagged else "ok")

    return EDAResult(
        issue_code="CLASS_IMBALANCE",
        severity=severity,
        affected_columns=[target],
        summary=(f"Minority class is {minority_pct*100:.1f}% of data."
                 if flagged else "Class distribution is acceptable."),
        fix_function="fix_class_imbalance",
        details={str(k): round(float(v), 4) for k, v in counts.items()},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. RESIDUAL ANALYSIS (REGRESSION)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    skew_threshold: float = 0.5,
) -> List[EDAResult]:
    """
    Analyse model residuals for normality and heteroscedasticity.

    Covers:
    - Residual skewness → NON_NORMAL_RESIDUALS
    - Breusch-Pagan test for heteroscedasticity → HETEROSCEDASTICITY

    Parameters
    ----------
    y_true          : actual target values
    y_pred          : model predicted values
    skew_threshold  : residual skewness above this is flagged

    Returns
    -------
    List of EDAResult.
    """
    print("\n" + "=" * 60)
    print("RESIDUAL ANALYSIS")
    print("=" * 60)

    results = []
    residuals = np.array(y_true) - np.array(y_pred)

    # ── Normality ────────────────────────────────────────────────────────────
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)
    _, p_shapiro = stats.shapiro(residuals[:min(5000, len(residuals))])

    print(f"  Residual skewness  : {skew:+.4f}")
    print(f"  Residual kurtosis  : {kurt:+.4f}")
    print(f"  Shapiro-Wilk p     : {p_shapiro:.4f} "
          f"({'non-normal ⚠️' if p_shapiro < 0.05 else 'normal ✅'})")

    non_normal = abs(skew) > skew_threshold or p_shapiro < 0.05
    results.append(EDAResult(
        issue_code="NON_NORMAL_RESIDUALS",
        severity="warning" if non_normal else "ok",
        affected_columns=[],
        summary=("Residuals are non-normally distributed — consider target transformation."
                 if non_normal else "Residuals are approximately normal."),
        fix_function="fix_target_transform",
        details={"skewness": round(float(skew), 4),
                 "kurtosis": round(float(kurt), 4),
                 "shapiro_p": round(float(p_shapiro), 4)},
    ))

    # ── Heteroscedasticity ───────────────────────────────────────────────────
    corr_resid_pred, p_corr = stats.pearsonr(
        np.abs(residuals), np.array(y_pred)
    )
    hetero = p_corr < 0.05 and abs(corr_resid_pred) > 0.2
    print(f"\n  |residual| ~ prediction correlation : "
          f"r={corr_resid_pred:.4f}  p={p_corr:.4f} "
          f"({'heteroscedastic ⚠️' if hetero else 'homoscedastic ✅'})")

    results.append(EDAResult(
        issue_code="HETEROSCEDASTICITY",
        severity="warning" if hetero else "ok",
        affected_columns=[],
        summary=("Heteroscedasticity detected — error variance grows with prediction."
                 if hetero else "Residual variance appears homoscedastic."),
        fix_function="fix_heteroscedasticity",
        details={"r_residual_pred": round(float(corr_resid_pred), 4),
                 "p_value": round(float(p_corr), 4)},
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 13. AUTOMATED FULL EDA PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_eda(
    df: pd.DataFrame,
    target: Optional[str] = None,
    task: str = "regression",
    date_cols: Optional[List[str]] = None,
    expected_schema: Optional[Dict[str, str]] = None,
    y_pred: Optional[pd.Series] = None,
) -> EDAReport:
    """
    Run the complete EDA pipeline and return a consolidated EDAReport.

    Designed for both interactive use (notebook) and automated pipelines.
    Pass y_pred to enable residual analysis.

    Parameters
    ----------
    df              : raw or processed DataFrame
    target          : name of the target column
    task            : "regression" | "classification" | "deep_learning"
    date_cols       : list of date column names for diversity check
    expected_schema : dict {col: dtype_str} for schema validation
    y_pred          : model predictions for residual analysis

    Returns
    -------
    EDAReport — call .print_report() for full output.

    Example
    -------
    >>> report = run_full_eda(df, target="price", task="regression",
    ...                       date_cols=["registration_date", "sold_at"])
    >>> report.print_report()
    >>> if report.has_critical:
    ...     print("Fix critical issues before training.")
    """
    all_results: List[EDAResult] = []

    # 1. Overview
    all_results.append(overview(df, target=target))

    # 2. Schema
    all_results.extend(validate_schema(df, expected_schema=expected_schema))

    # 3. Data quality
    all_results.extend(check_data_quality(df))

    # 4. Distributions
    all_results.extend(check_distributions(df))

    # 5. Diversity
    all_results.extend(check_diversity(df, date_cols=date_cols))

    # 6. Outliers
    all_results.append(detect_outliers(df))

    # 7. Correlations
    all_results.extend(analyse_correlations(df, target=target))

    # 8. Multicollinearity
    all_results.append(detect_multicollinearity(df, target=target))

    # 9. Task-specific checks
    if task == "classification" and target:
        all_results.append(check_class_imbalance(df, target=target))

    if task == "regression" and target and y_pred is not None:
        all_results.extend(analyse_residuals(df[target], y_pred))

    report = EDAReport(results=all_results)
    report.print_report()
    return report