from . import _check_func as Check
from . import core
from . import _configs as Config

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_numeric_boxplot(df: pd.DataFrame, numeric_cols: list[str], title_suffix: str = "", download_plot: bool = False):
    if not numeric_cols:
        print("⚠️ No valid numeric columns for boxplot.")
        return

    plot_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    spread_df = Check.distribution_report(df, numeric_cols, 1e-12, "fd").set_index("column")
    ordered_cols = spread_df.sort_values("spread_score", ascending=False).index.tolist()
    plot_df = plot_df[ordered_cols]

    fig_w = max(12, 0.6 * len(ordered_cols) + 4)
    fig_h = max(6, 0.35 * len(ordered_cols) + 2)

    plt.figure(figsize=(fig_w, fig_h))
    sns.boxplot(data=plot_df, orient="h", showfliers=True, linewidth=1)
    plt.title(f"Boxplot Numeric Columns {title_suffix}".strip())
    plt.xlabel("Value")
    plt.ylabel("Column")
    plt.tight_layout()
    if download_plot:
      plt.savefig(f"boxplot_numeric_columns{title_suffix.replace(' ', '_')}.png", dpi=300)
      return
    plt.show()

def plot_numeric_heatmap(df: pd.DataFrame, numeric_cols: list[str], title_suffix: str = "", download_plot: bool = False):
    if len(numeric_cols) < 2:
        print("⚠️ Heatmap requires at least 2 numeric columns.")
        return

    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().fillna(0)

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    annot = len(numeric_cols) <= 12

    fig_size = max(8, 0.55 * len(numeric_cols) + 4)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot=annot,
        fmt=".2f" if annot else None,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(f"Heatmap Correlation {title_suffix}".strip())
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if download_plot:
      plt.savefig(f"heatmap_correlation{title_suffix.replace(' ', '_')}.png", dpi=300)
      return
    plt.show()

def plots(df: pd.DataFrame, n_slice: int, download_plot: bool = False):
    if df.empty:
        print("⚠️ No data available for plotting.")
        return
    
    if df.shape[1] > 50 or (isinstance(n_slice, int) and n_slice > 50):
        core.WARNINGS.store(
            {
                "type": "UserWarning",
                "message": f"Too many features to be plotted ({df.shape[1]}), plot may be too ambiguous.",
                "where": "plots function"
            }
        )
        status = "n"
        if not Config.MUTE:
            status = input(f"Too many features to be plotted ({df.shape[1]}), plot may be too ambiguous. \nContinue with plotting? (y/n): ").strip().lower()
        
        if status == "n":
            return
        
        elif status == "y":
            pass

        else:
            print("Invalid input, skipping plotting process.")
            return

    if n_slice == "all":
        plot_numeric_boxplot(df, list(df.columns), title_suffix="(all features)", download_plot=download_plot)
        plot_numeric_heatmap(df, list(df.columns), title_suffix="(all features)", download_plot=download_plot)
        return

    n_slice = int(n_slice)
    cols = list(df.columns)

    for start in range(0, len(cols), n_slice):
        chunk_cols = cols[start:start + n_slice]
        chunk_df = df[chunk_cols]
        if chunk_df.empty or chunk_df.shape[1] < 2:
            continue
        title_suffix = f"(features {start + 1}-{start + len(chunk_cols)})"
        plot_numeric_boxplot(chunk_df, chunk_cols, title_suffix=title_suffix, download_plot=download_plot)
        plot_numeric_heatmap(chunk_df, chunk_cols, title_suffix=title_suffix, download_plot=download_plot)
