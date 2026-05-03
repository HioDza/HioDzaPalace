from . import _helper as Help
from . import _check_func as Check
from . import _plotting as PLT
from . import _info as Info
from . import _configs as Config

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import time
import json

from rich import print

# =============================
# Containers
# =============================
EXCEPTIONS = Help.Container("Exceptions", "Exceptions that occur during checks and analyses")
WARNINGS = Help.Container("Warnings", "Warnings that occur during checks and analyses")

# =============================
# Clarity score
# =============================
@dataclass
class CleanlinessBreakdown:
    missing_severity: float
    type_severity: float
    similarity_severity: float
    row_severity: float

    @property
    def overall(self) -> float:
        weights = {
            "missing_severity": 0.30,
            "type_severity": 0.20,
            "similarity_severity": 0.20,
            "row_severity": 0.30,
        }
        penalty = (
            self.missing_severity * weights["missing_severity"]
            + self.type_severity * weights["type_severity"]
            + self.similarity_severity * weights["similarity_severity"]
            + self.row_severity * weights["row_severity"]
        )
        return float(np.clip(1.0 - penalty, 0.0, 1.0))

    @property
    def label(self) -> str:
        score = self.overall
        if score >= 0.85:
            return "[green]very clean[/green]"
        if score >= 0.70:
            return "[yellow]fairly clean[/yellow]"
        if score >= 0.50:
            return "[orange1]some issues[/orange1]"
        return "[red]dirty[/red]"


def cleanliness_breakdown(
    nan_inf_df: pd.DataFrame,
    type_df: pd.DataFrame,
    sim_severity: float,
    row_severity: float,
) -> CleanlinessBreakdown:
    if len(nan_inf_df) == 0:
        missing_severity = 0.0
    else:
        missing_severity = float(np.clip(nan_inf_df["invalid_ratio"].mean(), 0.0, 1.0))

    if len(type_df) == 0:
        type_severity = 0.0
    else:
        type_severity = float(np.clip(type_df["bad_type_ratio"].mean(), 0.0, 1.0))

    return CleanlinessBreakdown(
        missing_severity=missing_severity,
        type_severity=type_severity,
        similarity_severity=float(np.clip(sim_severity, 0.0, 1.0)),
        row_severity=float(np.clip(row_severity, 0.0, 1.0)),
    )

# =============================
# Analysis
# =============================
def analyze(df: pd.DataFrame, 
             numeric_cols: list[str],
             target: str,
             task: str,
             json_output: bool = False) -> dict:
    sim_pairs, sim_cols, sim_severity = Check.abnormal_similarity_report(df, numeric_cols)
    nan_inf_df = Check.nan_inf_column_report(df, numeric_cols)
    type_df = Check.inconsistent_type_report(df, numeric_cols, thresh=0.05)
    structure_normality = Check.compute_normality(df, numeric_cols)
    row_df, row_scores, row_severity = Check.problematic_row_report(df, numeric_cols)
    dist_df = Check.distribution_report(df, numeric_cols)
    cleanliness = cleanliness_breakdown(nan_inf_df, type_df, sim_severity, row_severity)
    sparsity = Check.sparsity_ratio(df, numeric_cols)
    vif = Check.compute_vif(df, numeric_cols)
    linear_signal = Check.linear_signal(df, numeric_cols, target, task)

    if task == 'classification':
        override_rat = Check.class_override_ratio(df, numeric_cols, target)
        imbalance_rat = Check.class_imbalance_ratio(df, target)

    else:
        override_rat = None
        imbalance_rat = None

    shapiro = {c: Check.shapiro_per_feature(df[c]) for c in numeric_cols}
    ks = {c: Check.ks_per_feature(df[c]) for c in numeric_cols}
    
    if not json_output:
        return {
            "similarity_report": {
                "pairs": sim_pairs,
                "affected_columns": sim_cols,
                "severity": sim_severity,
            },
            "nan_inf_report": nan_inf_df,
            "type_inconsistency_report": type_df,
            "problematic_rows_report": {
                "rows": row_df,
                "severity": row_severity,
                "scores": row_scores,
            },
            "distribution_report": dist_df,
            "cleanliness_breakdown": cleanliness,
            "sparsity_ratio": sparsity,
            "vif": vif,
            "linear_signal": linear_signal,
            "class_override_ratio": override_rat,
            "class_imbalance_ratio": imbalance_rat,
            "structure_normality": structure_normality,
            "ks_scores": ks,
            "shapiro_scores": shapiro,
        }

    else:
        return {
            "feature_similarity_report": {
                "pairs": sim_pairs.to_dict(orient="records"),
                "affected_columns": sim_cols,
                "severity": sim_severity,
            },
            "nan_inf_report": nan_inf_df.head(10).to_dict(orient="records"),
            "type_inconsistency_report": type_df.head(10).to_dict(orient="records"),
            "problematic_rows_report": {
                "rows": row_df.head(10).to_dict(orient="records"),
                "severity": row_severity,
                "scores": row_scores.head(10).to_dict(),
            },
            "distribution_report": dist_df.head(10).to_dict(orient="records"),
            "cleanliness_breakdown": {
                "overall": cleanliness.overall,
                "label": cleanliness.label,
                "missing_severity": cleanliness.missing_severity,
                "type_severity": cleanliness.type_severity,
                "similarity_severity": cleanliness.similarity_severity,
                "row_severity": cleanliness.row_severity,
            },
            "sparsity_ratio": sparsity,
            "vif": {
                "mean": vif["mean"],
                "per_feature": {k: v if not np.isinf(v) else None for k, v in vif["per_feature"].items()},
            },
            "linear_signal": linear_signal,
            "class_override_ratio": override_rat,
            "class_imbalance_ratio": imbalance_rat,
            "structure_normality": structure_normality,
            "ks_scores": ks,
            "shapiro_scores": shapiro,
            "exceptions": EXCEPTIONS.data,
            "warnings": WARNINGS.data
        }

# =============================
# CLI
# =============================
def main():
    elapsed = time.time()
    parser = argparse.ArgumentParser(
        description="SanCheck — data sanity checker"
    )
    parser.add_argument(
                        "csv", 
                        help="Path to CSV file"
                        )
    
    parser.add_argument(
                        "target", 
                        help="Target column name"
                        )
    
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "auto"],
        default="auto",
        help="Type of machine learning task (default: auto)"
    )
    
    parser.add_argument(
        "--plot-chunk",
        default='all',
        type=Help.parse_slice_arg,
        help="Number of columns per chunk for plotting, or 'all' (default: 'all')",
    )
    
    parser.add_argument(
        "--download-plot",
        action="store_true",
        help="Download plots as PNG files (no plot visualization)",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (overrides --download-plot)",
    )

    parser.add_argument(
        "--metrics-info",
        action=Help.InfoAction,
        help="Show detailed explanation of metrics",
        nargs=0
    )

    parser.add_argument(
        "--get-json",
        action="store_true",
        help="Output the report as JSON instead of printing to console"
    )

    parser.add_argument(
        "--mute",
        action="store_true",
        help="Mute safeguards and automatically fallback to the safest option"
    )
    
    args = parser.parse_args()

    Config.MUTE = args.mute

    if args.metrics_info:
        Info.metrics()

    try:
        df = pd.read_csv(args.csv)

    except Exception as e:
        print(f"❌ Failed to load the CSV file: {e}")
        sys.exit(1)

    target = args.target
    if not isinstance(target, str):
        print(f"❌ Failed to process target. 'target' argument must be a string")
        sys.exit(1)

    elif target not in df.columns:
        print(f"❌ Failed to process target. {target} column is not exist in the DataFrame.")
        sys.exit(1)

    if args.task == "auto":
        task = Help._infer_task(df[target])

    else:
        task = args.task

    numeric_cols = Help.get_numeric_valid_columns(df)
    ignored_cols = [c for c in df.columns if c not in numeric_cols]

    if not numeric_cols:
        print("❌ No numeric columns found with sufficient valid numeric ratio.")
        print(f"Non-numeric columns / ignored: {', '.join(ignored_cols) if ignored_cols else '-'}")
        sys.exit(1)

    # JSON output
    get_json = args.get_json

    if get_json:
        with open("sancheck_report.json", "w") as f:
            json.dump(analyze(df, numeric_cols, target, task, True), f, cls=Help.ReportEncoder, indent=4)
        print("✅ Report saved to sancheck_report.json")
        # plotting
        if not args.no_plot:
            PLT.plots(df, n_slice=args.plot_chunk, download_plot=args.download_plot)
        return

    # Reports
    analysis_results = analyze(df, numeric_cols, target, task, False)

    sim_pairs = analysis_results["similarity_report"]["pairs"]
    sim_cols = analysis_results["similarity_report"]["affected_columns"]
    sim_severity = analysis_results["similarity_report"]["severity"]
    nan_inf_df = analysis_results["nan_inf_report"]
    type_df = analysis_results["type_inconsistency_report"]
    structure_normality = analysis_results["structure_normality"]
    linear_signal = analysis_results["linear_signal"]
    row_df = analysis_results["problematic_rows_report"]["rows"]
    row_scores = analysis_results["problematic_rows_report"]["scores"]
    row_severity = analysis_results["problematic_rows_report"]["severity"]
    dist_df = analysis_results["distribution_report"]
    cleanliness = analysis_results["cleanliness_breakdown"]
    sparsity = analysis_results["sparsity_ratio"]
    vif = analysis_results["vif"]
    override_rat = analysis_results["class_override_ratio"]
    imbalance_rat = analysis_results["class_imbalance_ratio"]
    shapiro = analysis_results["shapiro_scores"]
    ks = analysis_results["ks_scores"]

    # summary
    top_entropy = dist_df.sort_values("entropy", ascending=False).head(5)
    top_spread = dist_df.sort_values("spread_score", ascending=False).head(5)
    top_rows = row_df.head(5)
    
    # Colorful console report
    print("\n[bold cyan]📊 Dataset Summary[/bold cyan]")
    print(f"- Valid numeric columns: {len(numeric_cols)}")
    print(f"- Ignored non-numeric columns: {len(ignored_cols)}")
    print(f"- Inferred task type: {task}")

    print("\n[cyan]📌 Column problems[/cyan]")
    print(f"- Column with NaN/Inf/invalid: {len(nan_inf_df[nan_inf_df['invalid_ratio'] > 0])}")
    for _, r in nan_inf_df.sort_values("invalid_ratio", ascending=False).head(10).iterrows():
        print(
            f"  - {r['column']}: invalid={int(r['invalid_total'])}/{int(r['total'])} "
            f"({r['invalid_ratio']:.3f})"
        )

    print(f"- Type inconsistency column: {int(type_df['flagged'].sum())}")
    for _, r in type_df.sort_values("bad_type_ratio", ascending=False).head(10).iterrows():
        print(
            f"  - {r['column']}: bad_type={int(r['bad_type_total'])} "
            f"({r['bad_type_ratio']:.3f})"
        )

    if len(sim_pairs) > 0:
        print(f"- Similar feature pairs (|corr| >= {Config.DEFAULT_SIM_THRESHOLD}):")
        print(f"  - Severity similarity: {sim_severity:.3f}")
        for _, r in sim_pairs.head(10).iterrows():
            print(f"  - {r['col_a']} <-> {r['col_b']}: |corr|={r['abs_corr']:.3f}")
        if sim_cols:
            print(f"  - Affected columns: {', '.join(sim_cols)}")
    else:
        print("- Similar feature pairs: none")

    print("\n[cyan]📌 Row problems[/cyan]")
    invalid_row_count = int(row_df["has_invalid_numeric"].sum()) if len(row_df) else 0
    print(f"- Problematic rows (NaN/Inf): {invalid_row_count}/{len(df)}")
    print(f"- Severity row: {row_severity:.3f}")
    for _, r in top_rows.iterrows():
        print(
            f"  - row {r['row_index']}: score={r['row_anomaly_score']:.3f}, "
            f"invalid={bool(r['has_invalid_numeric'])}"
        )

    print("\n[cyan]📌 Distribution and interpretation[/cyan]")
    print("- High entropy means the distribution is more even/complex; it's not automatically 'noise', it can also be multimodal.")
    print("- High spread score means the data is more dispersed robustly compared to its central tendency.")
    print("\n  Top entropy (normalized 0-1):")
    for _, r in top_entropy.iterrows():
        print(
            f"  - {r['column']}: entropy={r['entropy']:.3f} "
            f"({r['entropy_label']})"
        )

    print("\n  Top spread (normalized 0-1):")
    for _, r in top_spread.iterrows():
        print(
            f"  - {r['column']}: spread_score={r['spread_score']:.3f} "
            f"({r['spread_label']}), var={r['variance']:.3f}, iqr={r['iqr']:.3f}"
        )
    
    print(f"\n[cyan]📌 Relation and structure[/cyan]")
    print(f"- VIF mean (normalized): {vif['mean']:.3f} ({Help._label_from_score(vif['mean'])})")
    print(f"- VIF per-feature")
    for c in numeric_cols:
        print(f"  - {c}: VIF={vif['per_feature'][c]:.3f}")
    print(f"- linear signal score: {linear_signal:.3f} ({Help._label_from_score(linear_signal, no_color=True)})")
    print(f"- sparsity: {sparsity:.3f} ({Help._label_from_score(sparsity)})")
    if task == 'classification':
        print(f"- class imbalance ratio: {imbalance_rat:.3f} ({Help._label_from_score(imbalance_rat)})")
        print(f"- class override ratio: {override_rat:.3f} ({Help._label_from_score(override_rat)})")

    print("\n[cyan]📌 Normality[/cyan]")
    print("- Shapiro-wilk and KS test score per-feature:")
    for c in numeric_cols:
        print(f"  - {c}: Shapiro={shapiro[c]:.3f} | KS={ks[c]:.3f}")
    print(f"- Structure normality (based on skewness and kurtosis): {structure_normality:.3f} ({Help._label_from_score(structure_normality, ascending=True)})")

    print("\n[cyan]🧼 Cleanineess status[/cyan]")
    print(f"- cleanliness score: {cleanliness.overall:.3f} / 1.000")
    print(f"- cleanliness label: {cleanliness.label}")
    print(f"- missing severity: {cleanliness.missing_severity:.3f}")
    print(f"- type severity: {cleanliness.type_severity:.3f}")
    print(f"- similarity severity: {cleanliness.similarity_severity:.3f}")
    print(f"- row severity: {cleanliness.row_severity:.3f}")

    print("\n[cyan]📊 Dataset-level distribution summary[/cyan]")
    print(f"- avg entropy: {dist_df['entropy'].mean():.3f}")
    print(f"- avg spread score: {dist_df['spread_score'].mean():.3f}")
    
    print(f"\n⏱️ Elapsed time: {time.time() - elapsed:.2f} seconds")
    print(f"\n❌ Exception(s) during process: {EXCEPTIONS.data}")
    print(f"\n⚠️ Warning(s) during process: {WARNINGS.data}")

    # plotting
    if not args.no_plot:
        PLT.plots(df, n_slice=args.plot_chunk, download_plot=args.download_plot)