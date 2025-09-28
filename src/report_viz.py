"""
Lightweight plotting helpers for the report notebook.
Usage in notebook:

    from src.report_viz import (
        set_theme,
        plot_kde_submissions_compact,
        plot_robust_box_pretty,
        plot_corr_heatmap_pretty,
    )
    set_theme()

"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def set_theme():
    import seaborn as sns  # optional

    try:
        sns.set_theme(style="whitegrid", context="talk", palette="deep")
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.figsize": (7.2, 3.6),
            "figure.dpi": 150,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )


def plot_kde_submissions_compact(subs: dict[str, str], clip_q=(0.001, 0.999), title="Submission pred distributions"):
    import seaborn as sns  # optional

    fig, ax = plt.subplots()
    lo_list, hi_list = [], []
    for name, path in subs.items():
        s = pd.read_parquet(path)["pred"].astype(float)
        lo, hi = s.quantile(clip_q[0]), s.quantile(clip_q[1])
        lo_list.append(lo)
        hi_list.append(hi)
        try:
            sns.kdeplot(s.clip(lo, hi), ax=ax, label=name, lw=2)
        except Exception:
            # fallback without seaborn
            from scipy.stats import gaussian_kde  # type: ignore

            xx = np.linspace(lo, hi, 400)
            kde = gaussian_kde(s.clip(lo, hi))
            ax.plot(xx, kde(xx), label=name, lw=2)
    if lo_list and hi_list:
        ax.set_xlim(min(lo_list), max(hi_list))
    ax.axvline(0, ls="--", c="k", lw=1, alpha=0.4)
    ax.set_title(title)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(subs), frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_robust_box_pretty(rob_df: pd.DataFrame):
    import seaborn as sns  # optional

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    try:
        sns.violinplot(data=rob_df, inner=None, cut=0, linewidth=0.8, saturation=0.9, ax=ax)
        sns.boxplot(
            data=rob_df,
            width=0.25,
            showcaps=True,
            boxprops={"zorder": 3, "alpha": 0.9},
            medianprops={"lw": 2, "color": "black"},
            ax=ax,
        )
    except Exception:
        rob_df.plot(kind="box", ax=ax)
    meds = rob_df.median()
    for i, k in enumerate([c for c in rob_df.columns]):
        ax.text(i, meds[k], f"{meds[k]:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Pearson r")
    if set(["eq", "grid", "ridge_clip"]).issubset(set(rob_df.columns)):
        ax.set_title(
            f"Holdout day-level resampling (r) | median: "
            f"eq={meds['eq']:.3f}  grid={meds['grid']:.3f}  ridge_clip={meds['ridge_clip']:.3f}"
        )
    else:
        ax.set_title("Holdout day-level resampling (r)")
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap_pretty(P: np.ndarray, labels: list[str] | None = None):
    import seaborn as sns  # optional

    C = np.corrcoef(P.T.astype(np.float64))
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    try:
        sns.heatmap(
            C,
            vmin=-1,
            vmax=1,
            center=0,
            cmap="coolwarm",
            square=True,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.75, "pad": 0.02},
            linewidths=0.3,
            linecolor="w",
            ax=ax,
        )
    except Exception:
        im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticklabels(labels)
    ax.set_title("Pairwise corr among models (holdout)")
    plt.tight_layout()
    plt.show()

