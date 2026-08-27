"""Build a frame-level, weighted training set for DeepTDA from TIS/RETIS
path-sampling trajectories.

Two dataset flavours are supported:

``prepare_deeptda_data`` (single simulation)
    Each output row is one phase point (frame) from a "plus ensemble" path,
    labeled with that path's reactive/non-reactive outcome (inherited by
    every frame of the path) and weighted by:

        frame_weight = path_weight * proximity_weight(op)

    ``path_weight`` is the WHAM-derived statistical weight of the path (see
    ``_compute_path_weights`` in ``shap_analysis.py``), unbiasing across the
    TIS ensembles.

    ``proximity_weight`` ramps from 0 (deep in stable state A, op <=
    lambda_low) to 1 (op >= lambda_high) via a smoothstep. Frames deep in A
    look the same in CV-space whether or not their path eventually reacted,
    so without this ramp the discriminant would be trained on a population
    with near-identical input but contradictory labels. Plus-ensemble paths
    terminate as soon as they cross the final interface, so there is no
    equivalent "deep in B" population to ramp down on the other end.

``prepare_deeptda_data_ld`` (two simulations, e.g. L/D enantiomers)
    Pools frames from two separate simulations (as in ``shap_analysis_ld.py``)
    and labels every frame by which simulation it came from (0/1) instead of
    by reactive/non-reactive outcome. Training a DeepTDA CV on the result
    (via ``train_deeptda``) yields a discriminant between the two
    simulations, and its input-correlation diagnostics reveal which
    collective variables drive the difference between them.
"""

import warnings
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import tomli
import typer

from inftools.tistools.shap_analysis import (
    _check_overwrite,
    _apply_angle_transforms,
    _apply_z_corrections,
    _compute_path_weights,
    _discover_columns,
    _extract_path_metadata,
    _load_path_table,
    _load_trajectory,
)
from inftools.tistools.shap_analysis_ld import (
    _PATHS_CHOICES,
    _apply_cv_rename,
    _load_interfaces_from_toml,
)

_Z_COLS = [
    "z_NTop", "z_NBot", "z_PTop", "z_PBot",
    "z_O2_T", "z_O2_B", "z_O3_T", "z_O3_B",
    "z_C2_T", "z_C2_B", "z_C3_T", "z_C3_B",
]


def _proximity_weight(op_vals, lambda_low, lambda_high):
    """Smoothstep ramp: 0 at/below lambda_low, 1 at/above lambda_high."""
    t = np.clip((op_vals - lambda_low) / (lambda_high - lambda_low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _subsample_frames(op_vals, cvs, stride, max_frames):
    """Stride and/or cap the frames kept from a single trajectory."""
    if stride > 1:
        op_vals = op_vals[::stride]
        cvs = cvs[::stride]
    if max_frames is not None and len(op_vals) > max_frames:
        idx = np.linspace(0, len(op_vals) - 1, max_frames).round().astype(int)
        op_vals = op_vals[idx]
        cvs = cvs[idx]
    return op_vals, cvs


def _frame_count_after_subsample(n_frames, stride, max_frames):
    """Frame count _subsample_frames would produce for a path with n_frames
    rows, without touching the actual data (mirrors its stride/cap logic)."""
    if stride > 1:
        n_frames = (n_frames + stride - 1) // stride  # len(arr[::stride])
    if max_frames is not None and n_frames > max_frames:
        n_frames = max_frames
    return n_frames


def _scan_trajectory(fpath, max_col, encoding):
    """Cheaply count a trajectory's data rows and check its column width,
    without parsing any floats (used to size the output arrays up front)."""
    n_rows = 0
    first_n_cols = None
    with fpath.open(encoding=encoding) as f:
        for i, line in enumerate(f):
            if i < 4:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if first_n_cols is None:
                first_n_cols = len(stripped.split())
            n_rows += 1
    if n_rows == 0 or first_n_cols is None or first_n_cols <= max_col:
        return None
    return n_rows


def _normalize_weights_mean1(w):
    """Rescale weights to mean 1 (no-op if already all-zero).

    Matches ``shap_analysis_ld``'s per-simulation normalisation: run this on
    each simulation's frame weights separately before pooling, so neither
    simulation dominates combined weighted statistics (e.g. the CV-vs-op
    correlation diagnostics in ``train_deeptda``) by its absolute weight
    scale.
    """
    mean = w.mean()
    return w / mean if mean > 0 else np.ones_like(w)


def _apply_flip(X, cv_names, flip_cols):
    """Negate (multiply by -1) the named CV columns in-place-equivalent copy.

    Use this to correct a systematic sign flip between two simulations -
    e.g. a dihedral angle that is a chirality-odd (pseudo-scalar) quantity,
    which is mathematically guaranteed to come out negated between mirror-
    image enantiomers for purely definitional reasons. Left uncorrected,
    such a CV trivially "perfectly" discriminates the two simulations
    without carrying any real information about differential behaviour,
    and dominates any downstream feature-importance ranking. Apply this to
    one simulation only (e.g. always -flip-cols-d) so the corrected values
    are directly comparable to the other simulation's raw values.
    """
    if not flip_cols:
        return X
    X = X.copy()
    name_to_idx = {name: i for i, name in enumerate(cv_names)}
    for col in flip_cols:
        if col not in name_to_idx:
            warnings.warn(
                f"-flip-cols: '{col}' not found in CV names {cv_names}; skipping.",
                stacklevel=2,
            )
            continue
        X[:, name_to_idx[col]] *= -1
        print(f"  Flipped sign of '{col}'")
    return X


def _apply_2d_transforms(X, cv_names, cos_cols, cos2_cols, z_cols):
    """Apply the angle/z-coordinate transforms to a flat (N_frames, N_cvs)
    frame matrix, via a size-1 trailing axis (those helpers were written for
    the (N_paths, N_cvs, N_grid) crossing-array shape used elsewhere)."""
    X3 = X[:, :, np.newaxis]
    X3, cv_names = _apply_angle_transforms(X3, cv_names, cos_cols, cos2_cols)
    X3, cv_names = _apply_z_corrections(X3, cv_names, z_cols)
    return X3[:, :, 0], cv_names


def _load_sim_frames(
    toml,
    data,
    cv_dir,
    nskip,
    op_col,
    cv_cols,
    stride,
    max_frames_per_path,
    paths_filter="all",
    lambda_low=None,
    lambda_high=None,
    force_proximity=False,
    encoding="utf-8",
    exclude=None,
):
    """Load one TIS/RETIS simulation's plus-ensemble frames into flat arrays.

    Every frame of every kept path is weighted by
    ``path_weight * proximity_weight(op)``. Proximity is flat 1.0 (no ramp)
    unless ``force_proximity`` is set or ``lambda_low``/``lambda_high`` are
    given explicitly - see the module docstring for why the single-simulation
    reactive/non-reactive dataset always needs the ramp, while the L/D
    dataset generally doesn't (every frame's label is unambiguous there).

    paths_filter : "all", "reactive", or "nonreactive" - which plus-ensemble
        paths to keep, by their reactive/non-reactive outcome.

    Returns
    -------
    X             : (N_frames, N_cvs) float32 CV matrix.
    w             : (N_frames,) float64 frame weight (path_weight * proximity).
    pnr_frame     : (N_frames,) int path number each frame belongs to.
    op_frame      : (N_frames,) float64 order-parameter value of each frame.
    reactive_frame: (N_frames,) float32, 1.0 = reactive path, 0.0 = non-reactive.
    cv_names      : list of str, length N_cvs.
    """
    cv_dir = Path(cv_dir)

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)

    use_proximity = force_proximity or lambda_low is not None or lambda_high is not None
    if use_proximity:
        if lambda_low is None:
            lambda_low = interfaces[0]
        if lambda_high is None:
            if M < 2:
                raise ValueError(
                    "Need >= 2 interfaces to default lambda_high; pass it explicitly."
                )
            lambda_high = interfaces[1]
        if lambda_high <= lambda_low:
            raise ValueError(
                f"lambda_high ({lambda_high}) must be > lambda_low ({lambda_low})."
            )

    pnr, maxop, path_f, path_w = _load_path_table(data, nskip, M)
    # WHAM weights need the full path set (both plus and minus ensembles
    # contribute to the cross-ensemble normalisation) - compute before
    # dropping minus-ensemble paths below.
    path_weights = _compute_path_weights(maxop, path_f, path_w, interfaces)

    labels_all, is_plus = _extract_path_metadata(cv_dir, pnr, encoding=encoding)

    # Only "plus ensemble" paths actually leave the A side; "minus ensemble"
    # paths never cross the interfaces and carry no DeepTDA signal.
    keep = is_plus & np.isfinite(labels_all)
    if paths_filter == "reactive":
        keep &= labels_all == 1.0
    elif paths_filter == "nonreactive":
        keep &= labels_all == 0.0
    n_dropped = len(pnr) - int(np.sum(keep))
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(pnr)} minus-ensemble/missing/filtered paths.")
    pnr, path_weights, labels = pnr[keep], path_weights[keep], labels_all[keep]

    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_dir, pnr, op_col, cv_cols, encoding, exclude=exclude
    )
    if exclude and cv_cols is None:
        print(f"Excluding CVs matching {exclude}: keeping {cv_names}")

    max_col = max(op_idx, max(cv_idxs, default=-1))

    # First pass: just count rows (cheap - no float parsing) so the output
    # arrays can be preallocated below. With thousands of long paths,
    # building per-path chunks and np.concatenate-ing them at the end would
    # briefly need 2x the final dataset's memory, which is what OOM-kills
    # this on large runs.
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
    total_frames = sum(frame_counts)

    # float32 for the CV matrix (by far the largest array) - plenty of
    # precision for ML training and halves its memory footprint.
    X = np.empty((total_frames, len(cv_idxs)), dtype=np.float32)
    w = np.empty(total_frames, dtype=np.float64)
    pnr_frame = np.empty(total_frames, dtype=pnr.dtype)
    op_frame = np.empty(total_frames, dtype=np.float32)
    reactive_frame = np.empty(total_frames, dtype=np.float32)

    offset = 0
    for row, n_kept in zip(rows, frame_counts):
        p = pnr[row]
        fpath = cv_dir / f"{p}.txt"
        frames = _load_trajectory(fpath, encoding)

        op_vals = frames[:, op_idx]
        cvs = frames[:, cv_idxs]
        op_vals, cvs = _subsample_frames(op_vals, cvs, stride, max_frames_per_path)

        if use_proximity:
            prox = _proximity_weight(op_vals, lambda_low, lambda_high)
        else:
            prox = 1.0
        frame_w = path_weights[row] * prox

        sl = slice(offset, offset + n_kept)
        X[sl] = cvs
        w[sl] = frame_w
        pnr_frame[sl] = p
        op_frame[sl] = op_vals
        reactive_frame[sl] = labels[row]
        offset += n_kept

    return X, w, pnr_frame, op_frame, reactive_frame, cv_names


def prepare_deeptda_data(
    toml: Annotated[str, typer.Option("-toml", help="Path to the infretis .toml config (for the TIS interfaces).")] = "infretis.toml",
    data: Annotated[str, typer.Option("-data", help="Path to the infretis_data.txt file.")] = "infretis_data.txt",
    cv_dir: Annotated[str, typer.Option("-cv-dir", help="Folder with per-path CV trajectory .txt files.")] = "ML",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of the data file (burn-in).")] = 1000,
    exclude: Annotated[str, typer.Option("-ex-cv", help="Comma-separated CV name(s) or substring(s) to exclude (e.g. a full column name or a shared prefix).")] = "",
    out: Annotated[str, typer.Option("-out", help="Output .npz path for the prepared dataset.")] = "deeptda_dataset.npz",
    op_col: Annotated[str, typer.Option("-op-col", help="Name of the order-parameter column in the CV files.")] = "OP_Lamb",
    lambda_low: Annotated[Optional[float], typer.Option("-lambda-low", help="Op value at/below which proximity weight is 0 (default: interfaces[0]).")] = None,
    lambda_high: Annotated[Optional[float], typer.Option("-lambda-high", help="Op value at/above which proximity weight is 1 (default: interfaces[1]).")] = None,
    stride: Annotated[int, typer.Option("-stride", help="Keep every Nth frame of each path.")] = 1,
    max_frames_per_path: Annotated[Optional[int], typer.Option("-max-frames-per-path", help="Cap frames kept per path (evenly subsampled); unset = no cap.")] = None,
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of files.")] = False,
):
    """Build a frame-level, path-weight x proximity-weight DeepTDA training
    set from TIS/RETIS trajectories, and save it to a compressed .npz."""
    _check_overwrite(out, overw)

    exclude_list = [p.strip() for p in exclude.split(",") if p.strip()] or None

    X, w, pnr_frame, op_frame, reactive_frame, cv_names = _load_sim_frames(
        toml=toml,
        data=data,
        cv_dir=cv_dir,
        nskip=nskip,
        op_col=op_col,
        cv_cols=None,
        stride=stride,
        max_frames_per_path=max_frames_per_path,
        paths_filter="all",
        lambda_low=lambda_low,
        lambda_high=lambda_high,
        force_proximity=True,
        exclude=exclude_list,
    )
    y = reactive_frame

    n_react = int(np.sum(y == 1.0))
    n_non = int(np.sum(y == 0.0))
    print(
        f"{len(np.unique(pnr_frame))} paths -> {len(y)} frames "
        f"({n_react} reactive / {n_non} non-reactive), {len(cv_names)} CVs."
    )

    np.savez_compressed(
        out,
        X=X,
        y=y,
        w=w,
        pnr=pnr_frame,
        op=op_frame,
        cv_names=np.array(cv_names),
    )
    print(f"Saved dataset to {out}")


def prepare_deeptda_data_ld(
    dir_l: Annotated[str, typer.Option("-dir-l", help="Root directory of the L simulation")] = "L",
    dir_d: Annotated[str, typer.Option("-dir-d", help="Root directory of the D simulation")] = "D",
    toml_l: Annotated[str, typer.Option("-toml-l", help="Toml filename inside -dir-l")] = "infretis.toml",
    data_l: Annotated[str, typer.Option("-data-l", help="Data filename inside -dir-l")] = "infretis_data.txt",
    toml_d: Annotated[str, typer.Option("-toml-d", help="Toml filename inside -dir-d")] = "infretis.toml",
    data_d: Annotated[str, typer.Option("-data-d", help="Data filename inside -dir-d")] = "infretis_data.txt",
    op_col: Annotated[str, typer.Option("-op-col", help="Name of the order-parameter column in the CV files.")] = "OP_Lamb",
    cv_cols: Annotated[Optional[str], typer.Option("-cv-cols", help="Comma-separated CV columns to use; default = all except -op-col")] = None,
    exclude: Annotated[Optional[str], typer.Option("-exclude", help="Comma-separated substrings; CVs whose name matches are dropped from both simulations")] = None,
    exclude_l: Annotated[Optional[str], typer.Option("-exclude-l", help="Comma-separated substrings; CVs whose name matches are dropped from the L simulation only")] = None,
    exclude_d: Annotated[Optional[str], typer.Option("-exclude-d", help="Comma-separated substrings; CVs whose name matches are dropped from the D simulation only")] = None,
    angle_cols: Annotated[Optional[str], typer.Option("-angle-cols", help="Comma-separated CV columns in degrees -> cos(theta)  [asymmetric molecule]")] = None,
    sym_angle_cols: Annotated[Optional[str], typer.Option("-sym-angle-cols", help="Comma-separated CV columns in degrees -> cos^2(theta)  [symmetric molecule]")] = None,
    flip_cols_l: Annotated[Optional[str], typer.Option("-flip-cols-l", help="Comma-separated CV columns to negate (x -> -x) in the L simulation only, before angle transforms. Use for chirality-odd CVs (e.g. dihedral angles) that are mathematically guaranteed to be sign-flipped between mirror-image enantiomers for purely definitional reasons; correcting one side makes the values directly comparable and stops the raw sign from trivially dominating the L/D discriminant.")] = None,
    flip_cols_d: Annotated[Optional[str], typer.Option("-flip-cols-d", help="Comma-separated CV columns to negate (x -> -x) in the D simulation only, before angle transforms. See -flip-cols-l.")] = None,
    name_cv_cols: Annotated[Optional[str], typer.Option("-name-cv-cols", help="Comma-separated 'old:new' pairs to normalise CV names, e.g. '_l_:_u_' or '_l_:_u_,foo:bar'. Applied to both simulations after angle transforms, before the compatibility check.")] = None,
    paths: Annotated[str, typer.Option("-paths", help="Which paths to include: 'all', 'reactive', 'nonreactive'")] = "all",
    nskip: Annotated[int, typer.Option("-nskip", help="Skip the first nskip rows of each infretis_data.txt (burn-in).")] = 1000,
    lambda_low: Annotated[Optional[float], typer.Option("-lambda-low", help="Op value at/below which proximity weight is 0; unset (with -lambda-high also unset) = no proximity ramp, every frame keeps its full path weight.")] = None,
    lambda_high: Annotated[Optional[float], typer.Option("-lambda-high", help="Op value at/above which proximity weight is 1; must be set together with -lambda-low.")] = None,
    stride: Annotated[int, typer.Option("-stride", help="Keep every Nth frame of each path.")] = 1,
    max_frames_per_path: Annotated[Optional[int], typer.Option("-max-frames-per-path", help="Cap frames kept per path (evenly subsampled); unset = no cap.")] = None,
    out: Annotated[str, typer.Option("-out", help="Output .npz path for the prepared dataset.")] = "deeptda_ld_dataset.npz",
    force_interfaces: Annotated[bool, typer.Option("-force-interfaces", help="Allow different interface counts between L and D. Unlike shap-enantiomer, no interpolation/common grid is needed here (frames are pooled raw, with no lambda feature) - this only relaxes the compatibility check; each simulation's own interfaces are still used independently for its own WHAM weights and proximity ramp.")] = False,
    overw: Annotated[bool, typer.Option("-O", help="Force overwriting of files.")] = False,
):
    """Build a frame-level DeepTDA training set that discriminates between two
    simulations (e.g. L vs D enantiomers of the same permeating molecule),
    labeled 0 (L) / 1 (D) instead of by reactive/non-reactive outcome.

    Every frame of every kept plus-ensemble path from both simulations is
    pooled, weighted by its (normalised) WHAM path weight. Feed the resulting
    .npz to ``train-deeptda`` to learn a 2-state DeepTDA CV that separates L
    from D; the printed/plotted correlations between that CV and each input
    CV (see ``train_deeptda``'s ``-class-names`` option) then show which
    collective variables carry the L/D difference.

    WHAM weights are normalised to mean 1 within each simulation before
    pooling (as in ``shap-enantiomer``) so neither simulation dominates by
    its absolute weight scale. Angle/z-coordinate transforms and CV name
    normalisation are applied exactly as in ``shap-enantiomer``, and the same
    CV-name compatibility check is enforced before pooling.
    """
    _check_overwrite(out, overw)

    if paths not in _PATHS_CHOICES:
        raise typer.BadParameter(f"paths={paths!r} — choose from {_PATHS_CHOICES}.")
    if (lambda_low is None) != (lambda_high is None):
        raise typer.BadParameter(
            "-lambda-low and -lambda-high must be set together (or both left unset)."
        )

    toml_l_path   = str(Path(dir_l) / toml_l)
    data_l_path   = str(Path(dir_l) / data_l)
    cv_dir_l_path = str(Path(dir_l) / "ML")
    toml_d_path   = str(Path(dir_d) / toml_d)
    data_d_path   = str(Path(dir_d) / data_d)
    cv_dir_d_path = str(Path(dir_d) / "ML")

    cv_cols_list   = cv_cols.split(",")        if cv_cols        else None
    cos_cols_list  = angle_cols.split(",")     if angle_cols     else None
    cos2_cols_list = sym_angle_cols.split(",") if sym_angle_cols else None
    flip_cols_l_list = flip_cols_l.split(",")  if flip_cols_l    else None
    flip_cols_d_list = flip_cols_d.split(",")  if flip_cols_d    else None
    _excl_shared = exclude.split(",")   if exclude   else []
    _excl_l_only = exclude_l.split(",") if exclude_l else []
    _excl_d_only = exclude_d.split(",") if exclude_d else []
    exclude_list_l = (_excl_shared + _excl_l_only) or None
    exclude_list_d = (_excl_shared + _excl_d_only) or None
    rename_pairs: list[tuple[str, str]] = []
    if name_cv_cols:
        for pair in name_cv_cols.split(","):
            old, sep, new = pair.partition(":")
            if not sep:
                raise typer.BadParameter(
                    f"-name-cv-cols: each substitution must be 'old:new', got {pair!r}"
                )
            rename_pairs.append((old, new))

    # ── Interface compatibility check (fast, no trajectory I/O) ────────────
    interfaces_l = _load_interfaces_from_toml(toml_l_path)
    interfaces_d = _load_interfaces_from_toml(toml_d_path)
    M_l, M_d = len(interfaces_l), len(interfaces_d)
    if M_l != M_d:
        if not force_interfaces:
            raise ValueError(
                f"Interface count mismatch: L has {M_l} interfaces, D has {M_d}. "
                "Use -force-interfaces to allow this (each simulation's own "
                "interfaces are still used independently for its WHAM weights "
                "and proximity ramp - no grid interpolation needed here)."
            )
        print(
            f"Interface count mismatch (L: {M_l}, D: {M_d}), proceeding because "
            "-force-interfaces was set. Each simulation still uses its own "
            "interfaces for WHAM weighting and proximity weighting."
        )
    if not np.isclose(interfaces_l[0], interfaces_d[0]) or not np.isclose(
        interfaces_l[-1], interfaces_d[-1]
    ):
        raise ValueError(
            f"Endpoint mismatch: L boundaries [{interfaces_l[0]}, {interfaces_l[-1]}] "
            f"vs D [{interfaces_d[0]}, {interfaces_d[-1]}]. "
            "First and last interfaces must agree."
        )

    print("Loading L simulation …")
    X_l, w_l, pnr_l, op_l, _, cv_names_l = _load_sim_frames(
        toml=toml_l_path, data=data_l_path, cv_dir=cv_dir_l_path, nskip=nskip,
        op_col=op_col, cv_cols=cv_cols_list, stride=stride,
        max_frames_per_path=max_frames_per_path, paths_filter=paths,
        lambda_low=lambda_low, lambda_high=lambda_high, exclude=exclude_list_l,
    )
    print("Loading D simulation …")
    X_d, w_d, pnr_d, op_d, _, cv_names_d = _load_sim_frames(
        toml=toml_d_path, data=data_d_path, cv_dir=cv_dir_d_path, nskip=nskip,
        op_col=op_col, cv_cols=cv_cols_list, stride=stride,
        max_frames_per_path=max_frames_per_path, paths_filter=paths,
        lambda_low=lambda_low, lambda_high=lambda_high, exclude=exclude_list_d,
    )

    # ── Sign correction (e.g. chirality-odd dihedrals), before angle/z
    # transforms so it acts on the raw degree-scale values ──────────────────
    X_l = _apply_flip(X_l, cv_names_l, flip_cols_l_list)
    X_d = _apply_flip(X_d, cv_names_d, flip_cols_d_list)

    # ── Angle transforms + z corrections (use original file column names) ──
    X_l, cv_names_l = _apply_2d_transforms(X_l, cv_names_l, cos_cols_list, cos2_cols_list, _Z_COLS)
    X_d, cv_names_d = _apply_2d_transforms(X_d, cv_names_d, cos_cols_list, cos2_cols_list, _Z_COLS)

    # ── CV name normalisation (e.g. _l_ → _u_) ─────────────────────────────
    if rename_pairs:
        cv_names_l = _apply_cv_rename(cv_names_l, rename_pairs)
        cv_names_d = _apply_cv_rename(cv_names_d, rename_pairs)

    # ── CV compatibility check ──────────────────────────────────────────────
    if cv_names_l != cv_names_d:
        raise ValueError(
            f"CV column mismatch between L and D (after name normalisation).\n"
            f"L: {cv_names_l}\nD: {cv_names_d}\n"
            "Use -name-cv-cols 'old:new' to normalise differing column names."
        )
    cv_names = cv_names_l

    # ── Normalise frame weights to mean 1 within each simulation ────────────
    w_l = _normalize_weights_mean1(w_l)
    w_d = _normalize_weights_mean1(w_d)

    X = np.vstack([X_l, X_d])
    y = np.concatenate(
        [np.zeros(len(X_l), dtype=np.float32), np.ones(len(X_d), dtype=np.float32)]
    )
    w = np.concatenate([w_l, w_d])
    op_frame = np.concatenate([op_l, op_d])
    pnr_frame = np.concatenate([pnr_l, pnr_d])
    sim = np.concatenate(
        [np.zeros(len(X_l), dtype=np.int8), np.ones(len(X_d), dtype=np.int8)]
    )

    print(
        f"L: {len(X_l)} frames  |  D: {len(X_d)} frames  |  "
        f"{len(cv_names)} CVs  |  paths='{paths}'"
    )

    np.savez_compressed(
        out,
        X=X,
        y=y,
        w=w,
        pnr=pnr_frame,
        op=op_frame,
        sim=sim,
        cv_names=np.array(cv_names),
    )
    print(f"Saved dataset to {out}")


if __name__ == "__main__":
    typer.run(prepare_deeptda_data)
