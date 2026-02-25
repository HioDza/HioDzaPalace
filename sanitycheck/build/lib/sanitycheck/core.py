#!/usr/bin/env python3
import argparse
import pandas as pd
import numpy as np
import math
import sys

# =============================
# Config (MVP defaults)
# =============================
NUMERIC_VALID_RATIO = 0.95
ENTROPY_BINS = 10

# =============================
# Utility
# =============================

def numeric_ratio(series: pd.Series):
    coerced = pd.to_numeric(series, errors="coerce")
    return coerced.notna().mean()


def get_numeric_valid_columns(df, thresh=NUMERIC_VALID_RATIO):
    return [
        c for c in df.columns
        if numeric_ratio(df[c]) >= thresh
    ]

# =============================
# Core Metrics
# =============================

def normalized_entropy(series, bins=ENTROPY_BINS):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return 0.0
    hist, _ = np.histogram(vals, bins=bins)
    probs = hist / hist.sum() if hist.sum() > 0 else []
    probs = probs[probs > 0]
    H = -(probs * np.log2(probs)).sum() if len(probs) > 0 else 0.0
    return float(H / math.log2(bins))


def normalized_variance(series, eps=1e-9):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) <= 1:
        return 0.0
    v = np.var(vals)
    mad = np.median(np.abs(vals - np.median(vals)))
    return float(v / (v + mad + eps))


# =============================
# Column Checks
# =============================

def abnormal_similarity(df: pd.DataFrame, numeric_cols: list, threshold=0.95):
    if len(numeric_cols) < 2:
        return []

    numeric = df[numeric_cols]
    corr = numeric.corr().abs()
    flagged = set()

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            if corr.iloc[i, j] >= threshold:
                flagged.add(numeric_cols[i])
                flagged.add(numeric_cols[j])

    return sorted(flagged)


def nan_inf_columns(df: pd.DataFrame, numeric_cols: list):
    cols = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any() or np.isinf(s.dropna()).any():
            cols.append(c)
    return cols


def inconsistent_type_columns(df: pd.DataFrame, numeric_cols: list, thresh=0.05):
    cols = []
    for c in numeric_cols:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        bad = coerced.isna() & s.notna()
        if bad.mean() > thresh:
            cols.append(c)
    return cols
    

def non_numeric_columns(df: pd.DataFrame, numeric_cols: list):
    return [c for c in df.columns if c not in numeric_cols]


# =============================
# Row Checks (FIXED)
# =============================

def problematic_rows(df: pd.DataFrame, numeric_cols: list):
    if not numeric_cols:
        return []

    coerced = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    mask = coerced.isna() | np.isinf(coerced)
    return list(df.index[mask.any(axis=1)])


# =============================
# Final Validation
# =============================

def final_validation(df, numeric_cols):
    if not numeric_cols:
        return 0.0, 0.0

    ent = [normalized_entropy(df[c]) for c in numeric_cols]
    var = [normalized_variance(df[c]) for c in numeric_cols]
    return float(np.mean(ent)), float(np.mean(var))


def clarity_score(df, sim_c, nan_c, type_c, row_c):
    C = max(1, df.shape[1])
    R = max(1, df.shape[0])

    w_sim, w_nan, w_type, w_row = 0.25, 0.30, 0.20, 0.15

    penalty = (
        (sim_c / C) * w_sim +
        (nan_c / C) * w_nan +
        (type_c / C) * w_type +
        (row_c / R) * w_row
    )
    penalty = min(1.0, penalty)
    return max(0.0, 1.0 - penalty)

def row_anomaly_scores(df, numeric_cols, eps=1e-9):
    if not numeric_cols:
        return pd.Series([], dtype=float)

    scores = []

    for c in numeric_cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        med = vals.median()
        mad = np.median(np.abs(vals - med))

        robust_z = (vals - med).abs() / (mad + eps)
        col_score = robust_z / (robust_z + 1)
        scores.append(col_score)

    score_df = pd.concat(scores, axis=1)
    return score_df.mean(axis=1)

# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(
        description="Data Sanity Checker — numeric-aware, no tuning"
    )
    parser.add_argument("csv", help="Path to CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        sys.exit(1)

    numeric_cols = get_numeric_valid_columns(df)

    sim_cols = abnormal_similarity(df, numeric_cols)
    nan_cols = nan_inf_columns(df, numeric_cols)
    type_cols = inconsistent_type_columns(df, numeric_cols)
    ignored_cols = non_numeric_columns(df, numeric_cols)
    rows = problematic_rows(df, numeric_cols)

    avg_entropy, avg_var = final_validation(df, numeric_cols)
    clarity = clarity_score(
        df,
        len(sim_cols),
        len(nan_cols),
        len(type_cols),
        len(rows),
    )

    col_problem_count = len(set(sim_cols + nan_cols + type_cols))

    row_scores = row_anomaly_scores(df, numeric_cols)
    top_rows = row_scores.sort_values(ascending=False).head(5)


    print(f"📌Column problem: [{col_problem_count}]")
    print(f"- [{', '.join(sim_cols) if sim_cols else '-'}]: Abnormal similiarity")
    print(f"- [{', '.join(nan_cols) if nan_cols else '-'}]: NaN/Inf")
    print(f"- [{', '.join(ignored_cols) if ignored_cols else '-'}]: Non-numeric (ignored)")
    print(f"- [{', '.join(type_cols) if type_cols else '-'}]: Inconsistent value type")

    print(f"📌Row problem: [{len(rows)}]")
    print(f"- [{', '.join(map(str, rows)) if rows else '-'}]: NaN/Inf (numeric columns only)")

    print("📌Top anomalous rows:")
    for idx, val in top_rows.items():
        print(f"- Row {idx}: score={val:.3f}")


    print("🔨 Final validation:")
    print(f"- [average entropy]: {avg_entropy:.3f}")
    print(f"- [average var]: {avg_var:.3f}")
    print(f"- clearity score: {clarity:.3f}")
