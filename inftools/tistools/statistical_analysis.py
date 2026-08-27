"""Direct feature-importance analysis of collective variables (CVs) for
TIS/RETIS path sampling, without machine-learning models.

For each TIS interface lambda_i (read from the infretis .toml file):
  1. Take every path's CV values at its first crossing of lambda_i.
  2. Label each path reactive / non-reactive (from its trajectory file header).
  3. Compute weighted effect-size metrics directly from WHAM-weighted data
     — no ML model is fitted, no train/test split, no cross-validation:
       • Cohen's d  — normalised mean separation between the two classes.
       • Spearman ρ — monotonic association of the CV value with the label.
       • KS distance — maximum gap between the two empirical weighted CDFs.
  4. Rank CVs by |Cohen's d| and produce plots analogous to the ML-SHAP output:
     a bar chart of effect sizes, a class strip plot (raw CV values coloured
     by reactivity), per-CV weighted histogram comparisons for the top CVs,
     and a cross-interface heatmap.

The full dataset contributes directly to every metric; nothing is held out.
"""

import datetime
import warnings
from pathlib import Path
from typing import Annotated, Optional

import matplotlib
import numpy as np
import tomli
import typer
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _check_overwrite(path, overw):
    if not overw and path and Path(path).exists():
        raise ValueError(f"Output file {path} already exists!")


# ---------------------------------------------------------------------------
# Weighted statistical metrics
# ---------------------------------------------------------------------------

def _weighted_mean_var(x, weights):
    """Weighted mean and variance."""
    w = weights / np.sum(weights)
    mu = np.dot(w, x)
    var = np.dot(w, (x - mu) ** 2)
    return mu, var


def _weighted_cohens_d(x, labels, weights):
    """Signed weighted Cohen's d (positive = reactive mean > non-reactive mean)."""
    mask1 = labels == 1
    mask0 = labels == 0
    if weights[mask1].sum() == 0 or weights[mask0].sum() == 0:
        return np.nan
    mu1, var1 = _weighted_mean_var(x[mask1], weights[mask1])
    mu0, var0 = _weighted_mean_var(x[mask0], weights[mask0])
    pooled_std = np.sqrt((var1 + var0) / 2.0)
    if pooled_std < 1e-12:
        return 0.0
    return (mu1 - mu0) / pooled_std


def _weighted_spearman(x, labels, weights):
    """Weighted Spearman ρ between x and binary labels (weighted Pearson on ranks)."""
    ranks = np.argsort(np.argsort(x)).astype(float)
    lab = labels.astype(float)
    w = weights / np.sum(weights)
    mu_r = np.dot(w, ranks)
    mu_l = np.dot(w, lab)
    cov = np.dot(w, (ranks - mu_r) * (lab - mu_l))
    std_r = np.sqrt(np.dot(w, (ranks - mu_r) ** 2))
    std_l = np.sqrt(np.dot(w, (lab - mu_l) ** 2))
    if std_r < 1e-12 or std_l < 1e-12:
        return 0.0
    return cov / (std_r * std_l)


def _weighted_ks(x, labels, weights):
    """Weighted KS distance between reactive and non-reactive distributions."""
    mask1 = labels == 1
    mask0 = labels == 0
    x1, w1 = x[mask1], weights[mask1]
    x0, w0 = x[mask0], weights[mask0]
    if w1.sum() == 0 or w0.sum() == 0:
        return np.nan

    def _ecdf(xs, ws, points):
        order = np.argsort(xs)
        ws_cum = np.cumsum(ws[order]) / ws.sum()
        idx = np.searchsorted(xs[order], points, side="right") - 1
        return np.where(idx >= 0, ws_cum[idx], 0.0)

    all_vals = np.sort(np.concatenate([x1, x0]))
    return float(np.max(np.abs(_ecdf(x1, w1, all_vals) - _ecdf(x0, w0, all_vals))))


def _compute_importance_metrics(X, y, weights, feature_names):
    """Compute Cohen's d, Spearman ρ, and KS distance per CV feature.

    Rows with any non-finite value are dropped per-feature independently.

    Returns a dict with arrays 'cohens_d', 'spearman', 'ks' (length N_cvs).
    """
    F = len(feature_names)
    cohens_d = np.full(F, np.nan)
    spearman = np.full(F, np.nan)
    ks_dist  = np.full(F, np.nan)

    for k in range(F):
        col = X[:, k]
        finite = np.isfinite(col) & np.isfinite(y) & np.isfinite(weights)
        if np.sum(finite) < 4:
            continue
        xf, yf, wf = col[finite], y[finite], weights[finite]
        if np.sum(yf == 1) == 0 or np.sum(yf == 0) == 0:
            continue
        cohens_d[k] = _weighted_cohens_d(xf, yf, wf)
        spearman[k] = _weighted_spearman(xf, yf, wf)
        ks_dist[k]  = _weighted_ks(xf, yf, wf)

    order = np.argsort(np.abs(cohens_d))[::-1]
    print(f"  {'rank':>4}  {'CV':<25}  {'Cohen d':>9}  {'Spearman r':>10}  {'KS':>7}")
    for rank, idx in enumerate(order, 1):
        print(
            f"  {rank:>4}  {feature_names[idx]:<25}  "
            f"{cohens_d[idx]:9.4f}  {spearman[idx]:10.4f}  {ks_dist[idx]:7.4f}"
        )

    return {"cohens_d": cohens_d, "spearman": spearman, "ks": ks_dist}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_effect_bar(effect_sizes, feature_names, out_path, overw=False):
    """Horizontal bar chart of |Cohen's d| — analogous to the SHAP bar plot."""
    _check_overwrite(out_path, overw)
    abs_eff = np.abs(effect_sizes)
    order = np.argsort(abs_eff)  # ascending so most important is at top
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(feature_names))))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, abs_eff[order], color="#1f77b4")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=9)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Feature Importance (|Cohen's d|)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Effect-size bar plot saved to {out_path}")


def _plot_class_strip(X, y, weights, feature_names, effect_sizes, out_path, overw=False):
    """Strip/jitter plot of normalised CV values per feature, coloured by class.

    Analogous to the SHAP beeswarm but shows raw CV values instead of SHAP
    contributions.  Features are ordered by |Cohen's d| (most important at top).
    """
    _check_overwrite(out_path, overw)
    n_cvs = len(feature_names)
    order = np.argsort(np.abs(effect_sizes))[::-1]

    col_min = np.nanmin(X, axis=0)
    col_max = np.nanmax(X, axis=0)
    X_norm = (X - col_min) / (col_max - col_min + 1e-12)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * n_cvs)))
    rng = np.random.default_rng(0)
    mask1 = y == 1
    mask0 = y == 0

    for row_pos, feat_idx in enumerate(order):
        col = X_norm[:, feat_idx]
        finite_mask = np.isfinite(col)
        n_fin = int(finite_mask.sum())
        jitter = rng.uniform(-0.2, 0.2, n_fin)

        fin_idx = np.where(finite_mask)[0]
        sel1 = mask1[fin_idx]
        sel0 = mask0[fin_idx]

        ax.scatter(
            col[fin_idx][sel1], row_pos + jitter[sel1],
            c="#d62728", alpha=0.3, s=4, rasterized=True,
        )
        ax.scatter(
            col[fin_idx][sel0], row_pos + jitter[sel0],
            c="#1f77b4", alpha=0.3, s=4, rasterized=True,
        )

    ax.set_yticks(range(n_cvs))
    ax.set_yticklabels([feature_names[i] for i in order], fontsize=8)
    ax.set_xlabel("Normalised CV value  (0 = min, 1 = max)")
    ax.set_title("CV distributions by class")
    ax.legend(
        handles=[
            Patch(color="#d62728", label="Reactive"),
            Patch(color="#1f77b4", label="Non-reactive"),
        ],
        fontsize=8, loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Class strip plot saved to {out_path}")


def _plot_cv_distributions(X, y, weights, feature_names, effect_sizes, out_dir, prefix, top_n, overw=False):
    """Weighted histogram comparison (reactive vs non-reactive) for top_n CVs."""
    order = np.argsort(np.abs(effect_sizes))[::-1]
    mask1 = y == 1
    mask0 = y == 0

    for rank, feat_idx in enumerate(order[:top_n]):
        fname = feature_names[feat_idx]
        safe = fname.replace("/", "_").replace(" ", "_")
        out_path = str(Path(out_dir) / f"{prefix}dist_{safe}.png")
        _check_overwrite(out_path, overw)

        col = X[:, feat_idx]
        fin1 = np.isfinite(col) & mask1
        fin0 = np.isfinite(col) & mask0

        x1, w1 = col[fin1], weights[fin1]
        x0, w0 = col[fin0], weights[fin0]
        if len(x1) == 0 or len(x0) == 0 or w1.sum() == 0 or w0.sum() == 0:
            continue

        xmin = min(x1.min(), x0.min())
        xmax = max(x1.max(), x0.max())
        bins = np.linspace(xmin, xmax, 31)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(x1, bins=bins, weights=w1 / w1.sum(), alpha=0.6,
                color="#d62728", label=f"Reactive (n={len(x1)})")
        ax.hist(x0, bins=bins, weights=w0 / w0.sum(), alpha=0.6,
                color="#1f77b4", label=f"Non-reactive (n={len(x0)})")
        d_val = effect_sizes[feat_idx]
        ax.set_xlabel(fname)
        ax.set_ylabel("Weighted fraction")
        ax.set_title(f"{fname}  (Cohen's d = {d_val:+.3f}, rank {rank + 1})")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Distribution plot saved to {out_path}")


def _plot_interface_heatmap_direct(results, cv_names, out_path, overw=False):
    """Heatmap of |Cohen's d| across all interfaces and CVs."""
    _check_overwrite(out_path, overw)
    valid = [r for r in results if r["ranking"] is not None]
    if not valid:
        return

    lambdas = [r["lambda"] for r in valid]
    cv_to_idx = {name: j for j, name in enumerate(cv_names)}
    mat = np.zeros((len(valid), len(cv_names)))
    for row_i, r in enumerate(valid):
        for cv_name, val in r["ranking"]:
            if cv_name in cv_to_idx:
                mat[row_i, cv_to_idx[cv_name]] = val

    fig, ax = plt.subplots(
        figsize=(max(6, 0.9 * len(cv_names)), max(4, 0.5 * len(valid)))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="|Cohen's d|")
    ax.set_xticks(np.arange(len(cv_names)))
    ax.set_xticklabels(cv_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(valid)))
    ax.set_yticklabels([f"λ={l:.4f}" for l in lambdas], fontsize=8)
    ax.set_xlabel("Collective Variable")
    ax.set_ylabel("Interface")
    ax.set_title("|Cohen's d| across interfaces")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Interface heatmap saved to {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def statistics(
    toml: Annotated[str, typer.Option("-toml", help="The .toml file")] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="The infretis_data.txt file")] = "infretis_data.txt",
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Path data folder with CV values in .txt files")] = "ML",
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column name")] = "OP_Lamb",
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CV columns to use; default = all except -op-col")] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name matches one are dropped (only when -cv-cols is unset)")] = None,
    angle_cols: Annotated[Optional[str], typer.Option("-angle-cols", help="Comma-separated CV columns in degrees to convert to cos(θ) (asymmetric molecules)")] = None,
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="Comma-separated CV columns in degrees to convert to cos²(θ) (symmetric molecules)")] = None,
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of infretis_data.txt")] = 1000,
    plot_dir: Annotated[str, typer.Option("-plot-dir", help="Root directory for output plots")] = "shap_plots",
    top_n: Annotated[int, typer.Option("-top-n", help="Number of top CVs for distribution comparison plots")] = 3,
    out: Annotated[str, typer.Option("-out", help="Output file for the CV rankings")] = "shap_wo_ml_ranking.txt",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of existing files")] = False,
):
    """Per-interface CV importance from raw WHAM-weighted data (no ML model).

    Three weighted effect-size metrics separate reactive from non-reactive paths
    at each TIS interface without fitting any classifier:

      Cohen's d  — normalised mean difference (positive = reactive > non-reactive).
      Spearman ρ — monotonic rank correlation with the binary reactive label.
      KS distance — maximum gap between the two empirical weighted CDFs.

    CVs are ranked by |Cohen's d|.  Output per interface (written to -plot-dir):
      interface_NNN_effect_bar.png  — bar chart of |Cohen's d|
      interface_NNN_strip.png       — strip plot (raw CV values, coloured by class)
      interface_NNN_dist_<CV>.png   — weighted histogram comparison for top-N CVs
      interface_heatmap.png         — |Cohen's d| across all interfaces and CVs

    Interfaces with fewer than 2 paths in either class are skipped.
    """
    from inftools.tistools.shap_analysis import (
        _load_path_table,
        _compute_path_weights,
        _extract_cv_crossings,
        _extract_path_metadata,
        _apply_angle_transforms,
    )

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    pnr, maxop, path_f, path_w = _load_path_table(data, nskip, M)
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)

    cv_array, cv_names, _ = _extract_cv_crossings(
        cv_dir=cv_dir,
        pnr_expected=pnr,
        subgrid=interfaces,
        op_col=op_col,
        cv_cols=cv_cols.split(",") if cv_cols else None,
        exclude=exclude.split(",") if exclude else None,
    )
    cv_array, cv_names = _apply_angle_transforms(
        cv_array, cv_names,
        cos_cols=angle_cols.split(",") if angle_cols else None,
        cos2_cols=sym_angle_cols.split(",") if sym_angle_cols else None,
    )
    labels, is_plus = _extract_path_metadata(cv_dir, pnr)

    cv_array      = cv_array[is_plus]
    labels        = labels[is_plus]
    path_weights  = path_weights[is_plus]

    N_cvs = len(cv_names)
    print(
        f"{int(np.sum(is_plus))} plus-ensemble paths  |  "
        f"{N_cvs} CVs  |  {M} interfaces."
    )

    _check_overwrite(out, overw)
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    results = []
    for i in range(M):
        start = datetime.datetime.now()
        X_i = cv_array[:, :, i]

        finite = np.all(np.isfinite(X_i), axis=1) & np.isfinite(labels)
        n_pos = int(np.sum(labels[finite] == 1))
        n_neg = int(np.sum(labels[finite] == 0))
        print(
            f"\n[interface {i + 1:3d}/{M}] lambda = {interfaces[i]:.4f}  "
            f"({n_pos} reactive, {n_neg} non-reactive)"
        )

        if min(n_pos, n_neg) < 2:
            print("  SKIP: fewer than 2 paths in one class.")
            results.append({"interface": i, "lambda": interfaces[i], "ranking": None})
            continue

        metrics = _compute_importance_metrics(X_i, labels, path_weights, cv_names)

        prefix = f"interface_{i:03d}_"
        _plot_effect_bar(
            metrics["cohens_d"], cv_names,
            str(Path(plot_dir) / f"{prefix}effect_bar.png"), overw=overw,
        )
        _plot_class_strip(
            X_i, labels, path_weights, cv_names, metrics["cohens_d"],
            str(Path(plot_dir) / f"{prefix}strip.png"), overw=overw,
        )
        _plot_cv_distributions(
            X_i, labels, path_weights, cv_names, metrics["cohens_d"],
            str(plot_dir), prefix=prefix, top_n=top_n, overw=overw,
        )

        abs_d = np.abs(metrics["cohens_d"])
        order = np.argsort(abs_d)[::-1]
        results.append({
            "interface": i,
            "lambda": interfaces[i],
            "ranking": [(cv_names[idx], float(abs_d[idx])) for idx in order],
            "metrics": metrics,
        })
        end = datetime.datetime.now()
        print(f"Interface {i + 1}/{M} done in {end - start}.")

    _plot_interface_heatmap_direct(
        results, cv_names,
        str(Path(plot_dir) / "interface_heatmap.png"), overw=overw,
    )

    with open(out, "w") as f:
        f.write(
            "# interface\tlambda\trank\tCV\t"
            "abs_cohens_d\tcohens_d\tspearman\tks\n"
        )
        for r in results:
            if r["ranking"] is None:
                continue
            m = r["metrics"]
            for rank, (cv_name, abs_d_val) in enumerate(r["ranking"], 1):
                k = cv_names.index(cv_name)
                f.write(
                    f"{r['interface']}\t{r['lambda']:.6f}\t{rank}\t{cv_name}\t"
                    f"{abs_d_val:.6f}\t{m['cohens_d'][k]:.6f}\t"
                    f"{m['spearman'][k]:.6f}\t{m['ks'][k]:.6f}\n"
                )
    print(f"\nRankings saved to {out}.")
