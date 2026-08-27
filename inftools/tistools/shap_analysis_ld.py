"""SHAP feature-importance analysis for L vs D enantiomer classification.

Given two TIS/RETIS simulations (L and D enantiomers of the same permeating
molecule), trains a WHAM-weighted classifier to predict which simulation each
path-interface crossing belongs to, then explains it with SHAP.

Design
------
Every (path, interface) crossing point from both simulations is pooled into
a single dataset with the interface value λᵢ appended as an extra feature.
The classification label is 0 (L) or 1 (D).  This means:

- A path that crosses M interfaces contributes M rows, one per interface.
- λ as a feature lets the model discover where along the reaction coordinate
  L and D differ most; the SHAP dependence plot for λ reveals this directly.
- StratifiedGroupKFold keeps all rows belonging to one path in the same fold,
  preventing leakage between train and test splits.

WHAM weights are normalised to mean 1 within each simulation before pooling,
so neither simulation dominates by its absolute weight scale.

Filtering (-paths):
  all         — all plus-ensemble paths (default)
  reactive    — only paths labelled reactive
  nonreactive — only paths labelled non-reactive

CLI command: ``inft shap-enantiomer``
"""

import datetime
import os
import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tomli
import typer

from .shap_analysis import (
    MODEL_CHOICES,
    _MODEL_LABELS,
    _apply_angle_transforms,
    _apply_z_corrections,
    _check_overwrite,
    _compute_path_weights,
    _extract_cv_crossings,
    _extract_path_metadata,
    _load_path_table,
    _model_shap_kfold,
    _plot_calibration,
    _plot_roc_curves,
    _plot_shap_bar,
    _plot_shap_dependence,
    _plot_shap_summary,
)

_PATHS_CHOICES = ("all", "reactive", "nonreactive")


def _load_interfaces_from_toml(toml_path):
    """Read only the interface list from a simulation toml (no trajectory I/O)."""
    with open(toml_path, "rb") as f:
        cfg = tomli.load(f)
    return np.asarray(cfg["simulation"]["interfaces"], dtype=float)


def _load_sim_data(
    toml_path, data_path, cv_dir, op_col, cv_cols_list, exclude_list, nskip,
    subgrid=None,
):
    """Load one TIS simulation into the arrays needed for L/D analysis.

    Parameters
    ----------
    subgrid : array-like or None
        Interface values at which to extract CV crossing points.  If None,
        the simulation's own interfaces are used.  Pass a common grid shared
        between L and D so that the lambda feature has identical values for
        both simulations and cannot confound the classifier.
        WHAM weights are always computed with the simulation's own interfaces.

    Returns
    -------
    interfaces   : (M,) float — simulation's own TIS interface values.
    pnr          : (N,) int  — path numbers.
    path_weights : (N,) float — WHAM path weights.
    cv_array     : (N, N_cvs, M_grid) float — CV values at first crossing.
    cv_names     : list[str] — CV column names, length N_cvs.
    labels       : (N,) float — 1 = reactive, 0 = non-reactive, NaN = missing.
    is_plus      : (N,) bool  — True for plus-ensemble paths.
    """
    with open(toml_path, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    pnr, maxop, path_f, path_w = _load_path_table(data_path, nskip, M)
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)

    crossing_grid = subgrid if subgrid is not None else interfaces
    cv_array, cv_names, _ = _extract_cv_crossings(
        cv_dir=cv_dir,
        pnr_expected=pnr,
        subgrid=crossing_grid,
        op_col=op_col,
        cv_cols=cv_cols_list,
        exclude=exclude_list,
    )
    labels, is_plus = _extract_path_metadata(cv_dir, pnr)

    return interfaces, pnr, path_weights, cv_array, cv_names, labels, is_plus


def _pool_ld_dataset(
    cv_array_l, path_weights_l, pnr_l, labels_l, is_plus_l,
    cv_array_d, path_weights_d, pnr_d, labels_d, is_plus_d,
    interfaces,
    paths_filter="all",
):
    """Pool L and D crossing points into a single (X, y, weights, groups) array.

    Each (path, interface) pair that exists becomes one row.  λᵢ is appended
    as the last feature column so the model can capture interface-specific
    L/D differences.  WHAM weights are normalised to mean 1 within each
    simulation before concatenation.  Group IDs are unique per path so that
    StratifiedGroupKFold keeps all rows from one path in the same fold.

    Returns
    -------
    X      : (N_rows, N_cvs + 1) — CV values at crossing + lambda.
    y      : (N_rows,) float     — 0 = L, 1 = D.
    w      : (N_rows,) float     — normalised WHAM weights.
    groups : (N_rows,) int       — unique per path for GroupKFold.
    """
    M = len(interfaces)
    X_parts, y_parts, w_parts, g_parts = [], [], [], []
    group_offset = 0

    for sim_id, (cv_array, path_weights, pnr, labels, is_plus) in enumerate([
        (cv_array_l, path_weights_l, pnr_l, labels_l, is_plus_l),  # 0 = L
        (cv_array_d, path_weights_d, pnr_d, labels_d, is_plus_d),  # 1 = D
    ]):
        sim_name = "L" if sim_id == 0 else "D"
        mask = is_plus & np.isfinite(labels)
        if paths_filter == "reactive":
            mask &= labels == 1.0
        elif paths_filter == "nonreactive":
            mask &= labels == 0.0

        cv_arr = cv_array[mask]      # (N, N_cvs, M)
        pw     = path_weights[mask]  # (N,)
        N      = cv_arr.shape[0]
        N_cvs  = cv_arr.shape[1]

        if N == 0:
            warnings.warn(
                f"Simulation {sim_name}: no paths remain after applying "
                f"paths='{paths_filter}'. Check your filter and data.",
                stacklevel=3,
            )
            group_offset += N
            continue

        # pw_norm = pw / pw.mean() if pw.mean() > 0 else np.ones(N) # Proportional to the number of paths
        pw_norm = pw

        # (N, N_cvs, M) → (N*M, N_cvs); tile interfaces to (N*M, 1)
        X_flat  = cv_arr.transpose(0, 2, 1).reshape(N * M, N_cvs)
        lam_col = np.tile(interfaces, N)[:, np.newaxis]
        X_parts.append(np.hstack([X_flat, lam_col]))

        y_parts.append(np.full(N * M, float(sim_id)))
        w_parts.append(np.repeat(pw_norm, M))
        # One unique group integer per path; offset avoids L/D collisions.
        g_parts.append(np.repeat(np.arange(N) + group_offset, M))
        group_offset += N

    if not X_parts:
        raise ValueError(
            "No paths remain in either simulation after filtering. "
            "Cannot build dataset."
        )

    return (
        np.vstack(X_parts),
        np.concatenate(y_parts),
        np.concatenate(w_parts),
        np.concatenate(g_parts).astype(int),
    )


def _apply_cv_negate(cv_array, cv_names, negate_substrings):
    """Negate CV columns whose name contains any of the given substrings.

    Operates on a copy of cv_array so the original is not mutated.  Use this
    to align dihedral-angle sign conventions between enantiomers: L has φ and
    D has −φ by symmetry, so negating one simulation's dihedrals puts both in
    the same conformational frame before the classifier sees them.

    Parameters
    ----------
    cv_array          : (N_paths, N_cvs, N_interfaces) float array.
    cv_names          : list of CV column name strings.
    negate_substrings : list of substrings; any CV whose name contains one is
                        negated.

    Returns
    -------
    cv_array : copy with selected columns multiplied by -1.
    cv_names : unchanged.
    """
    if not negate_substrings:
        return cv_array, cv_names
    cv_array = cv_array.copy()
    for i, name in enumerate(cv_names):
        if any(sub in name for sub in negate_substrings):
            cv_array[:, i, :] *= -1.0
    return cv_array, cv_names

def _apply_cv_angle_shift(cv_array, cv_names, shift_substrings):
    """Shift CV columns whose name contains any of the given substrings.

    Operates on a copy of cv_array so the original is not mutated.  Use this
    to align angles sign conventions between enantiomers where the difference 
    might be around 90 degrees.

    Parameters
    ----------
    cv_array          : (N_paths, N_cvs, N_interfaces) float array.
    cv_names          : list of CV column name strings.
    shift_substrings : list of substrings; any CV whose name contains one is
                       shifted.

    Returns
    -------
    cv_array : copy with selected columns shifted by 90 degrees.
    cv_names : unchanged.
    """
    if not shift_substrings:
        return cv_array, cv_names
    cv_array = cv_array.copy()
    for i, name in enumerate(cv_names):
        if any(sub in name for sub in shift_substrings):
            cv_array[:, i, :] -= 180.0
            cv_array[:, i, :] *= -1.0
    return cv_array, cv_names


def _monotone_nearest(shorter, longer):
    """For each element of `shorter` (sorted), return the nearest element of
    `longer` (sorted) while preserving order: no element of `longer` is reused
    and the selected indices are strictly increasing.

    Uses a greedy O(n + m) scan that is exact for sorted inputs.
    """
    matched = np.empty(len(shorter))
    j = 0
    for i, s in enumerate(shorter):
        # j_max: must leave enough elements in `longer` for the remaining matches
        j_max = len(longer) - (len(shorter) - i)
        while j < j_max and abs(longer[j + 1] - s) <= abs(longer[j] - s):
            j += 1
        matched[i] = longer[j]
        j += 1
    return matched


def _compute_common_grid(interfaces_l, interfaces_d):
    """Compute a common interface grid from two (possibly unequal-length) arrays.

    Equal counts: elementwise mean (unchanged from before).

    Unequal counts: the shorter array is taken as the skeleton; for each of its
    points the nearest point in the longer array is found via
    :func:`_monotone_nearest` (preserving order), and the pair is averaged.
    This keeps every common-grid point close to an actual interface position from
    both simulations, with no assumption about even spacing.

    Shared endpoints are pinned exactly so physical boundaries are not shifted
    by floating-point rounding.

    Returns
    -------
    common_grid : ndarray, shape (min(M_L, M_D),)
    """
    M_l, M_d = len(interfaces_l), len(interfaces_d)
    if M_l == M_d:
        grid = (interfaces_l + interfaces_d) / 2.0
    elif M_l < M_d:
        grid = (interfaces_l + _monotone_nearest(interfaces_l, interfaces_d)) / 2.0
    else:
        grid = (_monotone_nearest(interfaces_d, interfaces_l) + interfaces_d) / 2.0
    if np.isclose(interfaces_l[0],  interfaces_d[0]):
        grid[0]  = interfaces_l[0]
    if np.isclose(interfaces_l[-1], interfaces_d[-1]):
        grid[-1] = interfaces_l[-1]
    return np.round(grid, decimals=3)  # round to avoid floating-point noise


def _apply_cv_rename(cv_names, rename_pairs):
    """Apply substring substitutions to a list of CV names.

    rename_pairs is a list of (old, new) tuples applied in order.
    Substitutions that do not match any name are silently skipped.
    """
    result = list(cv_names)
    for old, new in rename_pairs:
        result = [name.replace(old, new) for name in result]
    return result


def shap_enantiomer(
    dir_l: Annotated[str, typer.Option("-dir-l", help="Root directory of the L simulation")] = "L",
    dir_d: Annotated[str, typer.Option("-dir-d", help="Root directory of the D simulation")] = "D",
    toml_l: Annotated[str, typer.Option("-toml-l", help="Toml filename inside -dir-l")] = "infretis.toml",
    data_l: Annotated[str, typer.Option("-data-l", help="Data filename inside -dir-l")] = "infretis_data.txt",
    toml_d: Annotated[str, typer.Option("-toml-d", help="Toml filename inside -dir-d")] = "infretis.toml",
    data_d: Annotated[str, typer.Option("-data-d", help="Data filename inside -dir-d")] = "infretis_data.txt",
    op_col: Annotated[str, typer.Option("-op-col", help="Order-parameter column name")] = "OP_Lamb",
    name_cv_cols: Annotated[Optional[str], typer.Option("-name-cv-cols", help="Comma-separated 'old:new' pairs to normalise CV names, e.g. '_l_:_u_' or '_l_:_u_,foo:bar'. Applied to both simulations after angle transforms, before the compatibility check.")] = None,
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CV columns to use; default = all except -op-col")] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name matches are dropped from both simulations")] = None,
    exclude_l: Annotated[Optional[str], typer.Option("-exclude-l", help="Comma-separated substrings; CVs whose name matches are dropped from the L simulation only")] = None,
    exclude_d: Annotated[Optional[str], typer.Option("-exclude-d", help="Comma-separated substrings; CVs whose name matches are dropped from the D simulation only")] = None,
    negate_l: Annotated[Optional[str], typer.Option("-negate-l", help="Comma-separated substrings; CVs whose name matches are multiplied by -1 in the L simulation (applied after z-corrections, before renaming)")] = None,
    negate_d: Annotated[Optional[str], typer.Option("-negate-d", help="Comma-separated substrings; CVs whose name matches are multiplied by -1 in the D simulation (applied after z-corrections, before renaming)")] = None,
    shift_l: Annotated[Optional[str], typer.Option("-shift-l", help="Comma-separated substrings; CVs whose name matches are shifted by +90 degrees in the L simulation (applied after z-corrections, before renaming)")] = None,
    shift_d: Annotated[Optional[str], typer.Option("-shift-d", help="Comma-separated substrings; CVs whose name matches are shifted by +90 degrees in the D simulation (applied after z-corrections, before renaming)")] = None,
    angle_cols: Annotated[Optional[str], typer.Option("-angle-cols", help="CV columns in degrees → cos(θ)  [asymmetric molecule]")] = None,
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="CV columns in degrees → cos²(θ)  [symmetric molecule]")] = None,
    paths: Annotated[str, typer.Option("-paths", help="Which paths to include: 'all', 'reactive', 'nonreactive'")] = "all",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip first nskip rows of each infretis_data.txt (equilibration)")] = 1000,
    models: Annotated[str, typer.Option("-models", help=f"Comma-separated models to run; choices: {', '.join(MODEL_CHOICES)}")] = "rf,gbm,lgbm,logreg,svm",
    n_splits: Annotated[int, typer.Option("-n-splits", help="Number of stratified group CV folds")] = 5,
    n_estimators: Annotated[int, typer.Option("-n-estimators", help="Trees per RandomForest / GradientBoosting")] = 300,
    n_jobs: Annotated[int, typer.Option("-n-jobs", help="CPU cores; -1 = all")] = 12,
    seed: Annotated[int, typer.Option("-seed", help="Random seed for fold splits and models")] = 42,
    plot_dir: Annotated[str, typer.Option("-plot-dir", help="Base name for root directory for plots; each model gets a subdirectory")] = "ld_plots",
    top_n: Annotated[int, typer.Option("-top-n", help="Top N CVs for SHAP dependence plots")] = 3,
    out: Annotated[str, typer.Option("-out", help="Base name for ranking files; '_<model>.txt' is appended")] = "shap_ld_ranking.txt",
    force_interfaces: Annotated[bool, typer.Option("-force-interfaces", help="Allow different interface counts between L and D; both grids are resampled to the shorter length via linear interpolation before averaging into a common grid")] = False,
    drop_z_ref: Annotated[bool, typer.Option("-drop-z-ref", help="Remove the z-reference column (z_Memb) from the feature set after z-corrections are applied; the column is still used as the reference during correction")] = False,
    optimize: Annotated[bool, typer.Option("-optimize", help="Run random hyperparameter search before the main k-fold loop; best params override -n-estimators and model defaults")] = False,
    n_search_iter: Annotated[int, typer.Option("-n-search-iter", help="Number of random hyperparameter configurations to evaluate when -optimize is set")] = 20,
    overw: Annotated[bool, typer.Option("-O", help="Overwrite existing files")] = False,
):
    """L vs D enantiomer SHAP analysis: which CVs distinguish the two simulations?

    Pools all (path, interface) crossing points from both simulations into one
    dataset and trains a classifier to predict whether each row belongs to L (0)
    or D (1).  The interface value λ is included as an extra feature so the
    model can capture where along the reaction coordinate the two enantiomers
    differ most.

    Cross-validation uses StratifiedGroupKFold: all crossing points of one path
    stay in the same fold, preventing data leakage between train and test.

    WHAM weights are normalised to mean 1 within each simulation so neither
    dominates the training by virtue of its absolute weight scale.

    High SHAP for a CV at a given λ means that CV value at that interface
    crossing is the clearest signal that the path belongs to L or D.
    """
    # ── Resolve full paths from directory + filename components ───────────
    toml_l_path   = str(Path(dir_l) / toml_l)
    data_l_path   = str(Path(dir_l) / data_l)
    cv_dir_l_path = str(Path(dir_l) / "ML")
    toml_d_path   = str(Path(dir_d) / toml_d)
    data_d_path   = str(Path(dir_d) / data_d)
    cv_dir_d_path = str(Path(dir_d) / "ML")

    if paths not in _PATHS_CHOICES:
        raise typer.BadParameter(f"paths={paths!r} — choose from {_PATHS_CHOICES}.")
    model_list = [m.strip() for m in models.split(",")]
    unknown = [m for m in model_list if m not in MODEL_CHOICES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown model(s): {unknown}. Available: {list(MODEL_CHOICES)}"
        )

    cv_cols_list   = cv_cols.split(",")        if cv_cols        else None
    cos_cols_list  = angle_cols.split(",")     if angle_cols     else None
    _excl_shared = exclude.split(",")   if exclude   else []
    _excl_l_only = exclude_l.split(",") if exclude_l else []
    _excl_d_only = exclude_d.split(",") if exclude_d else []
    exclude_list_l = (_excl_shared + _excl_l_only) or None
    exclude_list_d = (_excl_shared + _excl_d_only) or None
    cos2_cols_list = sym_angle_cols.split(",") if sym_angle_cols else None
    rename_pairs: list[tuple[str, str]] = []
    if name_cv_cols:
        for pair in name_cv_cols.split(","):
            old, sep, new = pair.partition(":")
            if not sep:
                raise typer.BadParameter(
                    f"-name-cv-cols: each substitution must be 'old:new', got {pair!r}"
                )
            rename_pairs.append((old, new))
    z_cols_list   = [
        "z_NTop", "z_NBot", "z_PTop", "z_PBot",
        "z_O2_T", "z_O2_B", "z_O3_T", "z_O3_B",
        "z_C2_T", "z_C2_B", "z_C3_T", "z_C3_B",
    ]

    # ── Interface compatibility check (fast, no trajectory I/O) ───────────
    interfaces_l_raw = _load_interfaces_from_toml(toml_l_path)
    interfaces_d_raw = _load_interfaces_from_toml(toml_d_path)
    M_l, M_d = len(interfaces_l_raw), len(interfaces_d_raw)
    if M_l != M_d:
        if not force_interfaces:
            raise ValueError(
                f"Interface count mismatch: L has {M_l} interfaces, D has {M_d}. "
                f"Use -force-interfaces to interpolate both to {min(M_l, M_d)} common points."
            )
    if not np.isclose(interfaces_l_raw[0], interfaces_d_raw[0]) or not np.isclose(
        interfaces_l_raw[-1], interfaces_d_raw[-1]
    ):
        raise ValueError(
            f"Endpoint mismatch: L boundaries [{interfaces_l_raw[0]}, {interfaces_l_raw[-1]}] "
            f"vs D [{interfaces_d_raw[0]}, {interfaces_d_raw[-1]}]. "
            "First and last interfaces must agree."
        )
    common_grid = _compute_common_grid(interfaces_l_raw, interfaces_d_raw)
    if M_l != M_d:
        print(
            f"Interface count mismatch (L: {M_l}, D: {M_d}). "
            f"Resampled to {len(common_grid)} common points via nearest-interface matching."
        )
        print(f"  L interfaces : {interfaces_l_raw.tolist()}")
        print(f"  D interfaces : {interfaces_d_raw.tolist()}")
        print(f"  Common grid  : {common_grid.tolist()}")
    elif np.max(np.abs(interfaces_l_raw - interfaces_d_raw)) > 1e-8:
        max_diff = float(np.max(np.abs(interfaces_l_raw - interfaces_d_raw)))
        print(
            f"Interface grids differ (max Δλ = {max_diff:.4f}). "
            "Using elementwise-mean common grid so λ values are identical for L and D."
        )
        print(f"  L interfaces : {interfaces_l_raw.tolist()}")
        print(f"  D interfaces : {interfaces_d_raw.tolist()}")
        print(f"  Common grid  : {common_grid.tolist()}")
    
    # ── Run file check for each model ─────────────────────────────────────────────────────
    # file check is done here before any heavy I/O or model training

    full_plot_dir = os.path.join(plot_dir, paths)
    Path(full_plot_dir).mkdir(parents=True, exist_ok=True)
    out_stem   = Path(out).stem
    out_suffix = Path(out).suffix
    if overw is False:        
        for model_type in model_list:
            model_plot_dir = Path(full_plot_dir) / model_type
            model_plot_dir.mkdir(parents=True, exist_ok=True)
            model_out = str(Path(out).parent / f"{out_stem}_{model_type}_{paths}{out_suffix}")
            _check_overwrite(model_out, overw)

    # ── Load both simulations using common grid for CV extraction ──────────
    print("Loading L simulation …")
    _, pnr_l, pw_l, cv_arr_l, cv_names_l, labels_l, is_plus_l = _load_sim_data(
        toml_l_path, data_l_path, cv_dir_l_path, op_col, cv_cols_list, exclude_list_l, nskip,
        subgrid=common_grid,
    )
    print("Loading D simulation …")
    _, pnr_d, pw_d, cv_arr_d, cv_names_d, labels_d, is_plus_d = _load_sim_data(
        toml_d_path, data_d_path, cv_dir_d_path, op_col, cv_cols_list, exclude_list_d, nskip,
        subgrid=common_grid,
    )

    interfaces = common_grid

    # ── Angle transforms (use original file column names from CLI) ─────────
    cv_arr_l, cv_names_l = _apply_angle_transforms(
        cv_arr_l, cv_names_l, cos_cols_list, cos2_cols_list
    )
    cv_arr_d, cv_names_d = _apply_angle_transforms(
        cv_arr_d, cv_names_d, cos_cols_list, cos2_cols_list
    )
    cv_arr_l, cv_names_l = _apply_z_corrections(cv_arr_l, cv_names_l, z_cols_list, drop_ref=drop_z_ref)
    cv_arr_d, cv_names_d = _apply_z_corrections(cv_arr_d, cv_names_d, z_cols_list, drop_ref=drop_z_ref)

    # ── Per-simulation sign flip (e.g. align dihedral conventions) ─────────
    negate_l_list = negate_l.split(",") if negate_l else []
    negate_d_list = negate_d.split(",") if negate_d else []
    cv_arr_l, cv_names_l = _apply_cv_negate(cv_arr_l, cv_names_l, negate_l_list)
    cv_arr_d, cv_names_d = _apply_cv_negate(cv_arr_d, cv_names_d, negate_d_list)

    # ── Per-simulation shift (e.g. align dihedral conventions) ─────────────
    shift_l_list = shift_l.split(",") if shift_l else []
    shift_d_list = shift_d.split(",") if shift_d else []
    cv_arr_l, cv_names_l = _apply_cv_angle_shift(cv_arr_l, cv_names_l, shift_l_list)
    cv_arr_d, cv_names_d = _apply_cv_angle_shift(cv_arr_d, cv_names_d, shift_d_list)

    # ── CV name normalisation (e.g. _l_ → _u_) ────────────────────────────
    # This solves name mismatch between L and D if the direction of the simulation was different
    if rename_pairs:
        cv_names_l = _apply_cv_rename(cv_names_l, rename_pairs)
        cv_names_d = _apply_cv_rename(cv_names_d, rename_pairs)

    # ── CV compatibility check ─────────────────────────────────────────────
    if cv_names_l != cv_names_d:
        raise ValueError(
            f"CV column mismatch between L and D (after name normalisation).\n"
            f"L: {cv_names_l}\nD: {cv_names_d}\n"
            "Use -name-cv-cols 'old:new' to normalise differing column names."
        )
    cv_names = cv_names_l

    feature_names = cv_names + ["lambda"]
    M = len(interfaces)

    print(
        f"L: {int(np.sum(is_plus_l))} plus-ensemble paths  |  "
        f"D: {int(np.sum(is_plus_d))} plus-ensemble paths  |  "
        f"{len(cv_names)} CVs + lambda  |  {M} interfaces  |  paths='{paths}'"
    )

    # ── Build pooled dataset ───────────────────────────────────────────────
    X, y, w, groups = _pool_ld_dataset(
        cv_arr_l, pw_l, pnr_l, labels_l, is_plus_l,
        cv_arr_d, pw_d, pnr_d, labels_d, is_plus_d,
        interfaces, paths_filter=paths,
    )
    print(
        f"Pooled: {int(np.sum(y == 0))} L rows + {int(np.sum(y == 1))} D rows "
        f"= {len(y)} total (NaN rows dropped inside k-fold)."
    )

    # ── Run each model ─────────────────────────────────────────────────────
    for model_type in model_list:
        label = _MODEL_LABELS[model_type]
        sep = "=" * 62
        print(f"\n{sep}\n  Model: {label}\n  Paths: {paths}\n{sep}\n")

        model_plot_dir = Path(full_plot_dir) / model_type
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        model_out = str(Path(out).parent / f"{out_stem}_{model_type}_{paths}{out_suffix}")

        start = datetime.datetime.now()
        shap_values, mean_abs_shap, fold_auc, fold_roc, oof_true, oof_proba = (
            _model_shap_kfold(
                X, y, feature_names,
                model_type=model_type,
                sample_weight=w,
                n_splits=n_splits,
                n_estimators=n_estimators,
                n_jobs=n_jobs,
                random_state=seed,
                groups=groups,
                optimize=optimize,
                n_search_iter=n_search_iter,
            )
        )
        print(f"Done in {datetime.datetime.now() - start}.")

        if shap_values is not None:
            _plot_shap_summary(
                shap_values, X, feature_names,
                str(model_plot_dir / "shap_beeswarm.png"), overw=overw,
            )
        _plot_shap_bar(
            mean_abs_shap, feature_names,
            str(model_plot_dir / "shap_bar.png"), overw=overw,
            xlabel="Mean permutation importance" if model_type == "svm" else "Mean |SHAP value|",
            title="Feature Importance (permutation)" if model_type == "svm" else "SHAP Feature Importance",
        )
        if shap_values is not None:
            _plot_shap_dependence(
                shap_values, X, feature_names, str(model_plot_dir),
                prefix="", top_n=top_n, overw=overw,
            )
        _plot_roc_curves(
            fold_roc, str(model_plot_dir / "roc.png"), overw=overw,
        )
        _plot_calibration(
            oof_true, oof_proba,
            str(model_plot_dir / "calibration.png"), overw=overw,
        )

        order = np.argsort(mean_abs_shap)[::-1]
        with open(model_out, "w") as f:
            f.write("# model\tpaths\trank\tfeature\tmean_abs_shap\tmean_fold_AUC\n")
            for rank, idx in enumerate(order, 1):
                f.write(
                    f"{model_type}\t{paths}\t{rank}\t"
                    f"{feature_names[idx]}\t{mean_abs_shap[idx]:.6f}\t"
                    f"{float(np.nanmean(fold_auc)):.4f}\n"
                )
        print(f"Ranking saved to {model_out}.")
