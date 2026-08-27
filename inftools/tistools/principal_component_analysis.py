"""Principal component analysis of collective variables (CVs) for TIS/RETIS
path sampling.

Loads the per-frame CV trajectories used elsewhere in this package (see
``shap_analysis.py``/``prepare_deeptda_data.py``) and runs a WHAM-weighted
PCA over them, to summarise the dominant modes of the configuration space
sampled across the TIS ensembles.

Optionally takes a *second* simulation (``-toml2``/``-data2``/``-cv-dir2``)
that shares the same CV columns - e.g. comparing two related systems.
The PCA basis is then fit jointly on both simulations' (weighted) frames,
with each simulation's weights renormalised to sum to 1 first so that neither
one dominates the basis purely by virtue of being more heavily sampled;
both simulations are then projected onto that shared basis so they can be
compared directly.

``-ensemble`` filters which paths go into that comparison (applied
identically to both simulations), e.g. ``-ensemble plus`` keeps only paths
that cross the interfaces. On top of that, this always runs three separate,
independent calculations one after another - all paths, reactive paths
only, and non-reactive paths only - writing three sets of output files
(``-out``/``-plot`` suffixed with ``_all``/``_reactive``/``_non-reactive``).

On alignment: the columns here are scalar CVs (distances, angles, order
parameters, ...) already computed per frame, not atomic Cartesian
coordinates, so there is nothing to rigid-body align - a CV is by
construction invariant to overall translation/rotation of the system. What
does matter is that CVs live on very different scales/units, so by default
each column is standardised (zero mean, unit variance, weighted by the same
WHAM path weights used for the PCA itself) before the decomposition;
otherwise a component dominated by, e.g., a distance in nm would swamp one
that is actually more informative but reported on a different scale.

Trajectory file layout (per path, e.g. ``ML/<path_nr>.txt``): see
``shap_analysis.py``.
"""

from pathlib import Path
from typing import Annotated, Optional

import matplotlib
import numpy as np
import tomli
import typer

matplotlib.use("Agg")  # non-interactive backend; safe for CLI use
import matplotlib.pyplot as plt

from inftools.tistools.prepare_deeptda_data import (
    _frame_count_after_subsample,
    _scan_trajectory,
    _subsample_frames,
)
from inftools.tistools.shap_analysis import (
    _check_overwrite,
    _compute_path_weights,
    _discover_columns,
    _extract_path_metadata,
    _load_path_table,
    _load_trajectory,
)

ENSEMBLE_CHOICES = ("all", "plus", "minus")
REACTIVE_SPLITS = ("all", "reactive", "non-reactive")


def _discover_paths(pnr, cv_dir, op_idx, cv_idxs, stride, max_frames_per_path, encoding):
    """Cheap first pass (no float parsing): which paths' files exist and
    have enough columns, and how many frames each contributes after
    subsampling - just enough to size the (small) output arrays up front.
    """
    max_col = max(op_idx, max(cv_idxs, default=-1))
    rows, raw_counts = [], []
    missing = 0
    for row, p in enumerate(pnr):
        fpath = cv_dir / f"{p}.txt"
        if not fpath.exists():
            missing += 1
            continue
        n_rows = _scan_trajectory(fpath, max_col, encoding)
        if n_rows is None:
            print(f"WARNING: {fpath} has too few columns, skipping.")
            continue
        rows.append(row)
        raw_counts.append(n_rows)
    if missing:
        print(f"WARNING: {missing}/{len(pnr)} trajectory files not found in {cv_dir}.")

    frame_counts = [
        _frame_count_after_subsample(n, stride, max_frames_per_path)
        for n in raw_counts
    ]
    return rows, frame_counts


def _iter_path_frames(
    pnr, path_weights, rows, frame_counts, cv_dir, op_idx, cv_idxs,
    stride, max_frames_per_path, encoding,
):
    """Yield (path_nr, path_weight, n_kept, op_vals, cvs) one path at a
    time, in the same order/subsampling as `rows`/`frame_counts` (from
    `_discover_paths`).

    This is a generator *function*, not a generator: calling it twice
    replays the identical frames in the identical order (the underlying
    subsampling is deterministic), so a moments pass and a projection pass
    over two separate calls stay perfectly consistent - without ever
    holding more than one path's frames (let alone the whole dataset's CV
    matrix) in memory at once. That matters here because thousands of
    paths x long trajectories x many CV columns can add up to several GB
    if materialised as one array, which is what was OOM-killing this
    before.
    """
    for row, n_kept in zip(rows, frame_counts):
        p = pnr[row]
        fpath = cv_dir / f"{p}.txt"
        frames = _load_trajectory(fpath, encoding)

        op_vals = frames[:, op_idx]
        cvs = frames[:, cv_idxs]
        op_vals, cvs = _subsample_frames(op_vals, cvs, stride, max_frames_per_path)

        yield p, path_weights[row], n_kept, op_vals, cvs


def _weighted_moments(path_iter, n_cvs):
    """Weighted mean and covariance of the CVs, accumulated in a single
    streaming pass over `path_iter` via the sufficient statistics
    sum(w*x) and sum(w*x x^T) - O(n_cvs^2) memory regardless of how many
    frames/paths are summed over."""
    s1 = np.zeros(n_cvs)
    s2 = np.zeros((n_cvs, n_cvs))
    wsum = 0.0
    for _, w_p, n_kept, _, cvs in path_iter:
        s1 += w_p * cvs.sum(axis=0)
        s2 += w_p * (cvs.T @ cvs)
        wsum += w_p * n_kept

    mean = s1 / wsum
    cov = s2 / wsum - np.outer(mean, mean)
    return mean, cov


def _pca_from_covariance(cov, n_components):
    """Eigendecomposition of a covariance (or correlation) matrix.

    Returns
    -------
    components               : (n_components, N_cvs) unit-norm eigenvectors
                                (loadings), sorted by decreasing eigenvalue.
    explained_variance_ratio : (n_components,) fraction of total variance.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    total_var = np.sum(eigvals)
    explained_variance_ratio = eigvals[:n_components] / total_var
    components = eigvecs[:, :n_components].T

    return components, explained_variance_ratio


def _prepare_simulation(
    toml, data, cv_dir, nskip, op_col, exclude_patterns, ensemble, reactive,
    reweight, stride, max_frames_per_path, encoding,
):
    """Load one simulation's path table, WHAM weights, CV columns and
    per-path frame counts - the per-simulation setup shared between
    single- and two-simulation PCA runs.

    Returns
    -------
    pnr, path_weights, cv_path, op_idx, cv_idxs, cv_names, rows, frame_counts
    """
    cv_path = Path(cv_dir)

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    pnr, maxop, path_f, path_w = _load_path_table(data, nskip, M)
    # WHAM weights need the full path set (both plus and minus ensembles
    # contribute to the cross-ensemble normalisation) - compute before any
    # ensemble/reactive filtering or reweight override below.
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)
    if not reweight:
        path_weights = np.ones_like(path_weights)

    if ensemble != "all" or reactive != "all":
        labels, is_plus = _extract_path_metadata(cv_path, pnr, encoding=encoding)
        keep = np.ones(len(pnr), dtype=bool)
        if ensemble != "all":
            keep &= is_plus if ensemble == "plus" else ~is_plus
        if reactive != "all":
            # labels is NaN for missing files, which compares False below
            # and so is dropped here too, same as everywhere else.
            keep &= labels == (1.0 if reactive == "reactive" else 0.0)
        n_dropped = len(pnr) - int(np.sum(keep))
        if n_dropped:
            print(
                f"Dropping {n_dropped}/{len(pnr)} paths not matching "
                f"ensemble={ensemble!r}/reactive={reactive!r} ({data})."
            )
        pnr, path_weights = pnr[keep], path_weights[keep]

    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_path, pnr, op_col, None, encoding, exclude=exclude_patterns
    )

    rows, frame_counts = _discover_paths(
        pnr, cv_path, op_idx, cv_idxs, stride, max_frames_per_path, encoding
    )

    return pnr, path_weights, cv_path, op_idx, cv_idxs, cv_names, rows, frame_counts


def _project_frames(frame_iter, mean, std_safe, standardize, components, label, sl_arrays, offset):
    """Project one simulation's frames onto an already-fit PCA basis,
    filling `sl_arrays` (scores, pnr_frame, op_frame, sim_frame) starting
    at `offset`. Returns the offset just past the last frame written."""
    scores, pnr_frame, op_frame, sim_frame = sl_arrays
    for p, _, n_kept, op_vals, cvs in frame_iter:
        centered = cvs - mean
        if standardize:
            centered = centered / std_safe
        sl = slice(offset, offset + n_kept)
        scores[sl] = centered @ components.T
        pnr_frame[sl] = p
        op_frame[sl] = op_vals
        sim_frame[sl] = label
        offset += n_kept
    return offset


def _suffixed_path(path_str, suffix):
    """Insert `_<suffix>` before a file's extension."""
    p = Path(path_str)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


def _run_pca(
    toml, data, cv_dir, toml2, data2, cv_dir2, two_sims, label1, label2,
    nskip, op_col, exclude_patterns, ensemble, reactive, n_components,
    standardize, reweight, stride, max_frames_per_path, encoding,
    out_path, plot_path,
):
    """One full PCA run (fit + project + save + plot) for a single
    all/reactive/non-reactive split."""
    pnr1, w1, cv_path1, op_idx1, cv_idxs1, cv_names, rows1, frame_counts1 = _prepare_simulation(
        toml, data, cv_dir, nskip, op_col, exclude_patterns, ensemble, reactive,
        reweight, stride, max_frames_per_path, encoding,
    )
    if exclude_patterns:
        print(f"Excluding CVs matching {exclude_patterns} -> {len(cv_names)} CVs kept.")

    pnr2 = w2 = cv_path2 = op_idx2 = cv_idxs2 = None
    rows2, frame_counts2 = [], []

    if two_sims:
        pnr2, w2, cv_path2, op_idx2, cv_idxs2, cv_names2, rows2, frame_counts2 = _prepare_simulation(
            toml2, data2, cv_dir2, nskip, op_col, exclude_patterns, ensemble, reactive,
            reweight, stride, max_frames_per_path, encoding,
        )
        if cv_names2 != cv_names:
            raise ValueError(
                "CV columns differ between the two simulations - PCA needs "
                f"the same CVs in the same order.\n  sim1: {cv_names}\n  sim2: {cv_names2}"
            )
        # Give each simulation equal say in the joint basis regardless of
        # how many paths/frames it has - otherwise the basis would just be
        # dominated by whichever simulation happens to be more heavily
        # sampled, rather than reflecting both equally.
        w1 = w1 / np.sum(w1)
        w2 = w2 / np.sum(w2)

    n_components = min(n_components, len(cv_names))

    def _frame_iter1():
        return _iter_path_frames(
            pnr1, w1, rows1, frame_counts1, cv_path1, op_idx1, cv_idxs1,
            stride, max_frames_per_path, encoding,
        )

    def _frame_iter2():
        return _iter_path_frames(
            pnr2, w2, rows2, frame_counts2, cv_path2, op_idx2, cv_idxs2,
            stride, max_frames_per_path, encoding,
        )

    def _frame_iter_all():
        yield from _frame_iter1()
        if two_sims:
            yield from _frame_iter2()

    total_frames1 = sum(frame_counts1)
    total_frames2 = sum(frame_counts2)
    total_frames = total_frames1 + total_frames2

    if total_frames == 0:
        print(f"No frames matched ensemble={ensemble!r}/reactive={reactive!r} - skipping.")
        return

    # Pass 1: weighted mean/covariance, accumulated path-by-path (and
    # simulation-by-simulation) - never holds the full (total_frames,
    # N_cvs) CV matrix in memory.
    mean, cov = _weighted_moments(_frame_iter_all(), len(cv_idxs1))
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    std_safe = np.where(std > 0, std, 1.0)

    decomp_matrix = cov / np.outer(std_safe, std_safe) if standardize else cov
    components, explained_variance_ratio = _pca_from_covariance(decomp_matrix, n_components)

    # Pass 2: project each simulation's (already-centred/scaled) frames
    # onto the shared components - only the small (total_frames,
    # n_components) result is kept, not the raw CV values.
    scores = np.empty((total_frames, n_components), dtype=np.float32)
    pnr_frame = np.empty(total_frames, dtype=pnr1.dtype)
    op_frame = np.empty(total_frames, dtype=np.float64)
    sim_frame = np.empty(total_frames, dtype=f"<U{max(len(label1), len(label2))}")
    arrays = (scores, pnr_frame, op_frame, sim_frame)

    offset = _project_frames(_frame_iter1(), mean, std_safe, standardize, components, label1, arrays, 0)
    if two_sims:
        offset = _project_frames(_frame_iter2(), mean, std_safe, standardize, components, label2, arrays, offset)

    if two_sims:
        print(
            f"{label1}: {len(rows1)} paths -> {total_frames1} frames | "
            f"{label2}: {len(rows2)} paths -> {total_frames2} frames | "
            f"{len(cv_names)} CVs, ensemble={ensemble!r}, reactive={reactive!r}."
        )
    else:
        print(
            f"{len(rows1)} paths -> {total_frames1} frames, "
            f"{len(cv_names)} CVs, ensemble={ensemble!r}, reactive={reactive!r}."
        )
    for i, evr in enumerate(explained_variance_ratio):
        top = np.argsort(np.abs(components[i]))[::-1][:3]
        top_str = ", ".join(f"{cv_names[j]} ({components[i][j]:+.2f})" for j in top)
        print(f"PC{i + 1}: {evr * 100:5.1f}% variance - top loadings: {top_str}")

    np.savez_compressed(
        out_path,
        scores=scores,
        explained_variance_ratio=explained_variance_ratio,
        components=components,
        mean=mean,
        std=std,
        cv_names=np.array(cv_names),
        pnr=pnr_frame,
        op=op_frame,
        sim=sim_frame,
    )
    print(f"Saved PCA results to {out_path}")

    if n_components < 2:
        print("Fewer than 2 components requested/available - skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    if two_sims:
        colors = {label1: "tab:blue", label2: "tab:orange"}
        for lbl in (label1, label2):
            m = sim_frame == lbl
            ax.scatter(scores[m, 0], scores[m, 1], s=4, alpha=0.6, label=lbl, color=colors[lbl])
        ax.legend()
        title = f"PCA of ({reactive}): {label1} vs {label2}"
    else:
        sca = ax.scatter(scores[:, 0], scores[:, 1], c=op_frame, cmap="viridis", s=4, alpha=0.6)
        fig.colorbar(sca, ax=ax, label=op_col)
        title = f"PCA of ({reactive})"
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}%)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA scatter plot to {plot_path}")


def PCA(
    toml: Annotated[str, typer.Option("-toml", help="Path to the infretis .toml config (for the TIS interfaces).")] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="Path to the infretis_data.txt file.")] = "infretis_data.txt",
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Folder with per-path CV trajectory .txt files.")] = "ML",
    toml2: Annotated[Optional[str], typer.Option("-toml2", help="Optional second simulation's infretis .toml, to fit a joint PCA basis over both and compare them (must share the same CVs as the first).")] = None,
    data2: Annotated[Optional[str], typer.Option("-data2", help="Second simulation's infretis_data.txt (required together with -toml2).")] = None,
    cv_dir2: Annotated[Optional[str], typer.Option("-cv-dir2", help="Second simulation's CV trajectory folder. Required together with -toml2/-data2 - not defaulted to -cv-dir, since the two simulations commonly share the same relative folder name (e.g. both have an 'ML' subfolder) and silently reusing -cv-dir would read sim1's files for both.")] = None,
    label1: Annotated[str, typer.Option("-label1", help="Legend label for the first simulation (only used when -toml2/-data2 are given).")] = "sim1",
    label2: Annotated[str, typer.Option("-label2", help="Legend label for the second simulation.")] = "sim2",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of the data file(s) (burn-in).")] = 1000,
    op_col: Annotated[str, typer.Option("-op-col", help="Name of the order-parameter column in the CV files (used for plot colouring).")] = "OP_Lamb",
    exclude: Annotated[str, typer.Option("-ex-cv", help="Comma-separated CV name(s) or substring(s) to exclude.")] = "",
    ensemble: Annotated[str, typer.Option("-ensemble", help="Which paths to include: 'all', 'plus' (cross the interfaces) or 'minus' (sample below lambda_0).")] = "plus",
    n_components: Annotated[int, typer.Option("-n-components", help="Number of principal components to keep.")] = 5,
    standardize: Annotated[bool, typer.Option("-standardize/-no-standardize", help="Standardise each CV (weighted zero mean/unit variance) before PCA; disable to run PCA on the raw CV scales.")] = True,
    reweight: Annotated[bool, typer.Option("-reweight/-no-reweight", help="Weight frames by their path's WHAM weight, unbiasing across the TIS ensembles; disable to weight every frame equally.")] = True,
    stride: Annotated[int, typer.Option("-stride", help="Keep every Nth frame of each path.")] = 1,
    max_frames_per_path: Annotated[Optional[int], typer.Option("-max-frames-per-path", help="Cap frames kept per path (evenly subsampled); unset = no cap.")] = None,
    out: Annotated[str, typer.Option("-out", help="Output .npz path for the PCA results (suffixed _all/_reactive/_non-reactive for the three runs).")] = "pca_results.npz",
    plot: Annotated[str, typer.Option("-plot", help="Output .png path for the PC1-vs-PC2 scatter plot (suffixed _all/_reactive/_non-reactive for the three runs).")] = "pca_scatter.png",
    encoding: Annotated[str, typer.Option("-encoding", help="Text encoding of the .txt trajectory files.")] = "utf-8",
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of files.")] = False,
):
    """WHAM-weighted PCA over the per-frame CVs of TIS/RETIS path-sampling
    trajectories. Always runs three separate, independent calculations -
    all paths, reactive paths only, and non-reactive paths only - one after
    another, writing three suffixed sets of output files. Optionally fits
    a joint basis over a second simulation (-toml2/-data2/-cv-dir2) sharing
    the same CVs, to compare the two directly in the same PC space."""
    if ensemble not in ENSEMBLE_CHOICES:
        raise ValueError(f"-ensemble must be one of {ENSEMBLE_CHOICES}, got {ensemble!r}")

    two_sims = toml2 is not None or data2 is not None or cv_dir2 is not None
    if two_sims and (toml2 is None or data2 is None or cv_dir2 is None):
        raise ValueError(
            "-toml2, -data2 and -cv-dir2 must all be given together to enable "
            "two-simulation mode (-cv-dir2 is required explicitly, not "
            "defaulted to -cv-dir, to avoid silently reading sim1's CV files "
            "for sim2)."
        )
    cv_dir2_eff = cv_dir2

    out_paths = {r: _suffixed_path(out, r) for r in REACTIVE_SPLITS}
    plot_paths = {r: _suffixed_path(plot, r) for r in REACTIVE_SPLITS}
    for r in REACTIVE_SPLITS:
        _check_overwrite(out_paths[r], overw)
        _check_overwrite(plot_paths[r], overw)

    exclude_patterns = [p.strip() for p in exclude.split(",") if p.strip()]

    for reactive in REACTIVE_SPLITS:
        print(f"=== reactive={reactive!r} ===")
        _run_pca(
            toml, data, cv_dir, toml2, data2, cv_dir2_eff, two_sims, label1, label2,
            nskip, op_col, exclude_patterns, ensemble, reactive, n_components,
            standardize, reweight, stride, max_frames_per_path, encoding,
            out_paths[reactive], plot_paths[reactive],
        )


if __name__ == "__main__":
    typer.run(PCA)
