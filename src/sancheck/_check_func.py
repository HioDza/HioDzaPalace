from . import _helper as Help
from . import _configs as Config
from . import core

import pandas as pd
import numpy as np
import math
from scipy import stats

from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif, f_regression, mutual_info_classif, mutual_info_regression

import warnings

# =============================
# Distribution analysis
# =============================
def normalized_entropy(series: pd.Series, eps: float, bins):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) < 2:
        return 0.0

    if np.all(vals == vals[0]):
        return 0.0

    try:
        hist, edges = np.histogram(vals, bins=bins)

    except Exception as e:
        core.EXCEPTIONS.store({
                "type": type(e).__name__,
                "message": str(e),
                "where": "normalized_entropy computation",
            })
        hist, edges = np.histogram(vals, bins=min(10, max(2, int(np.sqrt(len(vals))))))

    total = hist.sum()
    if total <= 0:
        return 0.0

    probs = hist / total
    probs = probs[probs > 0]
    if len(probs) <= 1:
        return 0.0

    H = -(probs * np.log2(probs)).sum()
    H_max = math.log2(len(hist)) if len(hist) > 1 else 1.0
    return float(np.clip(H / max(H_max, eps), 0.0, 1.0))


def normalized_spread_score(series: pd.Series, eps: float):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0, 0.0, 0.0

    var = float(np.var(vals, ddof=1)) if len(vals) > 1 else 0.0
    q75, q25 = np.percentile(vals, [75, 25])
    iqr = float(q75 - q25)
    robust_sigma = iqr / 1.349 if iqr > 0 else float(np.std(vals, ddof=1))
    baseline = robust_sigma ** 2

    score = var / (var + baseline + eps)
    score = float(np.clip(score, 0.0, 1.0))
    return score, var, iqr


def distribution_report(df: pd.DataFrame, 
                        numeric_cols: list[str], 
                        eps: float=Config.EPS, 
                        bins=Config.ENTROPY_BINS):
    rows = []
    for c in numeric_cols:
        ent = normalized_entropy(df[c], eps, bins)
        spread_score, raw_var, iqr = normalized_spread_score(df[c], eps)
        rows.append({
            "column": c,
            "entropy": ent,
            "entropy_label": Help.entropy_interpretation(ent),
            "spread_score": spread_score,
            "spread_label": Help.spread_interpretation(spread_score),
            "variance": raw_var,
            "iqr": iqr,
        })
    return pd.DataFrame(rows)

def class_override_ratio(df: pd.DataFrame, numeric_cols: list[str], target: str):
    cols = list(set(numeric_cols + [target]))
    sub = df[cols].dropna()

    grouped = sub.groupby(numeric_cols)[target].nunique()

    conflict = (grouped > 1).sum()
    total = len(grouped)

    if total == 0:
        return 0.0

    return conflict / total

def class_imbalance_ratio(df: pd.DataFrame, target: str):
    unique_classes = df[target].dropna().unique()
    if len(unique_classes) <= 1:
        return 0.0
    
    if len(unique_classes) > 50:
        core.WARNINGS.store({
            "type": "UserWarning",
            "message": f"Too many unique classes in target ({len(unique_classes)}), imbalance ratio may be less meaningful.",
            "where": "class_imbalance_ratio computation"
        })
        status = "n"
        if not Config.MUTE:
            status = input(f"⚠️ Too many unique classes in target ({len(unique_classes)}), imbalance ratio may be less meaningful.\nContinue with imbalance ratio calculation? (y/n): ").strip().lower()
        if status == 'n':
            return 0.0

        elif status == 'y':
            print("Continuing with imbalance ratio calculation...")
        
        else:
            print("Invalid input, skipping imbalance ratio calculation.")
            return 0.0

    y = df[target].dropna().to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y)

    max_label = y.max() if len(y) > 0 else 0
    counts = np.bincount(y, minlength=max_label + 1)
    probs = counts / len(y)
    gini = 1.0 - np.sum(probs ** 2)
    balance_rat = gini / (1 - 1 / len(unique_classes))
    
    return 1.0 - balance_rat

# =============================
# Column problems
# =============================
def nan_inf_column_report(df: pd.DataFrame, numeric_cols: list[str]):
    rows = []
    for c in numeric_cols:
        s = df[c]
        coerced, finite_mask, nan_mask, bad_parse_mask = Help._to_numeric_with_mask(s)

        total = len(s)
        non_null = int(s.notna().sum())
        invalid_total = int((~finite_mask).sum())  # nan + inf setelah coercion
        inf_total = int(np.isinf(coerced.to_numpy(dtype="float64", copy=False)).sum()) if non_null else 0
        nan_total = int(coerced.isna().sum())
        bad_parse_total = int(bad_parse_mask.sum())

        severity = invalid_total / max(total, 1)

        rows.append({
            "column": c,
            "total": total,
            "non_null": non_null,
            "nan_total": nan_total,
            "inf_total": inf_total,
            "bad_parse_total": bad_parse_total,
            "invalid_total": invalid_total,
            "invalid_ratio": float(severity),
        })
    return pd.DataFrame(rows)


def inconsistent_type_report(df: pd.DataFrame, 
                             numeric_cols: list[str], 
                             thresh: float=0.05):
    rows = []
    for c in numeric_cols:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        bad = coerced.isna() & s.notna()
        ratio = float(bad.mean())
        rows.append({
            "column": c,
            "bad_type_total": int(bad.sum()),
            "bad_type_ratio": ratio,
            "flagged": ratio > thresh,
        })
    return pd.DataFrame(rows)


def abnormal_similarity_report(df: pd.DataFrame, 
                               numeric_cols: list[str], 
                               threshold: float=Config.DEFAULT_SIM_THRESHOLD):
    if len(numeric_cols) < 2:
        return pd.DataFrame(), [], 0.0

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().abs()

    pairs = []
    flagged_cols = set()
    over_threshold_scores = []

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    pairs_df = corr.where(mask).stack().reset_index()
    pairs_df.columns = ["col_a", "col_b", "abs_corr"]
    pairs = pairs_df[pairs_df["abs_corr"] >= threshold][["col_a", "col_b", "abs_corr"]].values.tolist()
    flagged_cols = set(pairs_df[pairs_df["abs_corr"] >= threshold][["col_a", "col_b"]].values.ravel())
    over_threshold_scores = pairs_df[pairs_df["abs_corr"] >= threshold]["abs_corr"].tolist()

    total_pairs = len(pairs)
    issue_pairs = sum(1 for _, _, v in pairs if v >= threshold)

    pair_ratio = issue_pairs / max(total_pairs, 1)
    excess_mean = float(np.mean(over_threshold_scores)) if over_threshold_scores else 0.0

    severity = float(np.clip(0.6 * pair_ratio + 0.4 * excess_mean, 0.0, 1.0))
    report = pd.DataFrame(pairs, columns=["col_a", "col_b", "abs_corr"]).sort_values("abs_corr", ascending=False)

    return report, sorted(flagged_cols), severity


# =============================
# Row problems
# =============================
def problematic_row_report(df: pd.DataFrame, numeric_cols: list[str], eps: float=Config.EPS):
    if not numeric_cols:
        return pd.DataFrame(), pd.Series(dtype=float), 0.0

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(numeric.to_numpy(dtype="float64", copy=False))
    invalid_row_mask = invalid_mask.any(axis=1)

    col_scores = []
    for c in numeric_cols:
        vals = numeric[c]
        med = vals.median(skipna=True)
        mad = np.median(np.abs(vals.dropna() - med)) if vals.notna().any() else 0.0
        robust_z = (vals - med).abs() / (mad + eps)
        col_scores.append(robust_z / (robust_z + 1.0))

    if col_scores:
        score_df = pd.concat(col_scores, axis=1)
        row_scores = score_df.mean(axis=1, skipna=True).fillna(0.0)
    else:
        row_scores = pd.Series(np.zeros(len(df)), index=df.index)

    invalid_ratio = float(invalid_row_mask.mean()) if len(df) else 0.0
    anomaly_mean = float(row_scores.mean()) if len(row_scores) else 0.0
    severity = float(np.clip(0.7 * invalid_ratio + 0.3 * anomaly_mean, 0.0, 1.0))

    out = pd.DataFrame({
        "row_index": df.index,
        "has_invalid_numeric": invalid_row_mask,
        "row_anomaly_score": row_scores.values,
    })
    out = out.sort_values("row_anomaly_score", ascending=False)

    return out, row_scores, severity

# =============================
# Relation and sparsity
# =============================
def compute_vif(df: pd.DataFrame, numeric_cols: list[str]):
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 2:
        return 0.0
    
    vif_scores = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        try:
            for i in range(df.shape[1]):
                vif = variance_inflation_factor(df.values, i)

                if np.isinf(vif) or np.isnan(vif):
                    vif_scores.append(float('inf'))

                vif_scores.append(vif)
        
        except Exception as e:
            core.EXCEPTIONS.store({
            "type": type(e).__name__,
            "message": str(e),
            "where": "compute_vif computation",
        })
            vif_scores.append(float('inf'))
            
        for warn in w:
            core.WARNINGS.store({
            "type": warn.category.__name__,
            "message": str(warn.message),
            "where": "compute_vif computation",
        })

    raw = np.mean(vif_scores)
    norm_vif = np.tanh(raw / 10)

    return {
    "mean": norm_vif,
    "per_feature": dict(zip(numeric_cols, vif_scores))
    }

def linear_signal(df: pd.DataFrame, 
                  numeric_cols: list[str], 
                  target: str, 
                  task: str, 
                  eps: float=Config.EPS) -> float:
    cols = list(set(numeric_cols + [target]))
    df = df[cols].dropna()
    if df.shape[0] < 2:
        return 0.0

    X = df[numeric_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    y = df[target].to_numpy()
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            if task == "classification":
                f_scores, _ = f_classif(X, y)
                mean_f_score = np.mean(f_scores)
                score = np.tanh(mean_f_score / 10)

                mi = mutual_info_classif(X, y, discrete_features='auto')
                dependency = np.mean(mi)

                final_score = score / (dependency + eps)

                return float(np.clip(final_score, 0.0, 1.0))

            elif task == "regression":
                f_scores, _ = f_regression(X, y)
                mean_f_score = np.mean(f_scores)
                score = np.tanh(mean_f_score / 10)

                mi = mutual_info_regression(X, y, discrete_features='auto')
                dependency = np.mean(mi)

                final_score = score / (dependency + eps)

                return float(np.clip(final_score, 0.0, 1.0))

        except Exception as e:
            core.EXCEPTIONS.store({
                "type": type(e).__name__,
                "message": str(e),
                "where": "linear_signal computation",
            })

        for warn in w:
            core.WARNINGS.store({
                "type": warn.category.__name__,
                "message": str(warn.message),
                "where": "linear_signal computation",
            })
    return 0.0

def sparsity_ratio(df: pd.DataFrame, numeric_cols: list[str]):
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    n_samples, n_features = df.shape

    zero_ratio = np.sum(df == 0) / df.size
    dim_penalty = n_features / (n_samples + n_features)
    
    return 0.7 * zero_ratio + 0.3 * dim_penalty

# =============================
# Normality
# =============================
def shapiro_per_feature(series: pd.Series):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(vals)
    
    if n < 3:
        return 0.0
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            if n <= 5000:
                _, p = stats.shapiro(vals)
                return float(p)
            
            rng = np.random.default_rng(42)
            sample = rng.choice(vals, size=5000, replace=False)

            _, p = stats.shapiro(sample)
            return float(p)
        
        except Exception as e:
            core.EXCEPTIONS.store({
                "type": type(e).__name__,
                "message": str(e),
                "where": "shapiro_per_feature computation",
            })

        for warn in w:
            core.WARNINGS.store({
                "type": warn.category.__name__,
                "message": str(warn.message),
                "where": "shapiro_per_feature computation",
            })

    return 0.0


def ks_per_feature(series: pd.Series, eps: float=Config.EPS):
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(vals) < 3:
        return 0.0

    std = np.std(vals)
    if std <= eps:
        return 1.0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            _, p = stats.kstest(vals, "norm", args=(np.mean(vals), std))
            return float(p)

        except Exception as e:
            core.EXCEPTIONS.store({
                "type": type(e).__name__,
                "message": str(e),
                "where": "ks_per_feature computation",
            })

        for warn in w:
            core.WARNINGS.store({
                "type": warn.category.__name__,
                "message": str(warn.message),
                "where": "ks_per_feature computation",
            })

    return 0.0

def compute_normality(df: pd.DataFrame, numeric_cols: list[str]):
    df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if df.shape[1] == 0:
        return 0.5

    skew_vals = []
    kurt_vals = []

    for col in df.columns:
        if df[col].nunique() > 1:
            skew_vals.append(abs(skew(df[col].dropna())))
            kurt_vals.append(abs(kurtosis(df[col].dropna())))

    skew_mean = np.mean(skew_vals) if skew_vals else 0.0
    kurt_mean = np.mean(kurt_vals) if kurt_vals else 0.0

    # Normalize (heuristic scaling)
    skew_score = np.tanh(skew_mean / 2)
    kurt_score = np.tanh(kurt_mean / 5)

    normality = 1 - (0.5 * skew_score + 0.5 * kurt_score)

    return float(np.clip(normality, 0.0, 1.0))