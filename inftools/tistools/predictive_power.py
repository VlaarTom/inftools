"""
RETIS/TIS path sampling analysis: crossing probability, PDF,
and overlap integral / predictive power (T, S) for collective variable assessment.

Notation (from the paper):
  lambda_i       : TIS interfaces (sorted ascending), i = 0, ..., M-1
  lambda^c       : "current" crossing surface (fine-grid index alpha)
  lambda^r       : "reactive" target surface  (fine-grid index beta)
  t_q            : fraction of trajs crossing bin q at lambda^c
  r_q            : fraction of reactive trajs crossing bin q (reach lambda^r)
  u_q            : fraction of unreactive trajs crossing bin q (don't reach lambda^r)
  T              : predictive power = 1 - S
  S              : overlap integral of r and u distributions

Algorithm reference sections:
  I.   Crossing probability  v(alpha) = P(lambda^alpha | lambda_0)
  II.  Probability distribution functions  M_r, M_u
  III. Overlap S and predictive power T
"""

import os
import numpy as np
import tomli
from scipy.signal import savgol_filter
from typing import Annotated
import typer
from pathlib import Path
import warnings
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; safe for CLI use
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize


def _check_overwrite(path, overw):
    if not overw and path and os.path.exists(path):
        raise ValueError(f"Output file {path} already exists!")


def compute_crossing_probability(
    toml,
    data,
    out,
    outP,
    nskip,
    overw,
    n_subinterfaces,
    ):

    """Compute WHAM crossing probability P(lambda | lambda_0) on a fine grid.

    Algorithm (from paper, section "Numerical results - Crossing Probability"):
      1. Initialise v(alpha) = 0 for all fine-grid sub-interfaces alpha.
      2. For every trajectory X in every ensemble i:
           a. Determine lambda_max(X).
           b. For each alpha where lambda_i <= lambda^alpha < lambda_max(X),
              increment v(alpha) by the raw path weight w_{i,X}.
      3. For each alpha, determine K(alpha) (the highest TIS interface index
         that is <= lambda^alpha) and Q_{K(alpha)}, then multiply:
              v(alpha) = v(alpha) * Q_{K(alpha)}

    The Q_k normalisation factors are built from the WHAM crossing probabilities
    at each TIS interface, accumulated iteratively:
        P(lambda_1 | lambda_0)   = n_0(lambda_1) / n_0
        P(lambda_{i+1}|lambda_0) = sum_j n_j(lambda_{i+1})
                                   / sum_j n_j / P(lambda_j | lambda_0)

    Parameters
    ----------
    toml            : path to infretis .toml config (contains interface positions)
    data            : path to infretis data file (path_nr, len, maxop, pf..., pw...)
    out             : path to save per-trajectory weights  (path_nr, maxop, weight)
    outP            : optional path to save (lambda, P(lambda|lambda_0)) table
    nskip           : number of initial rows to discard (burn-in)
    overw           : if False, raise if output files already exist
    n_subinterfaces : number of fine-grid points spanning [lambda_0, lambda_max]

    Returns
    -------
    subgrid   : (n_subinterfaces,) fine lambda grid
    v         : (n_subinterfaces,) P(lambda^alpha | lambda_0)
    path_data : dict with keys pnr, maxop, weights (per-trajectory)
    """
    _check_overwrite(out, overw)
    _check_overwrite(outP, overw)

    # Load config and raw data
    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M = len(interfaces)  # number of TIS ensembles

    raw = np.loadtxt(data, dtype=str)
    raw = raw[nskip:]

    # Paths with "----" in column 3 are 0+ paths, skip rest
    non_zero = raw[:, 3] != "----"
    raw[raw == "----"] = "0.0"

    pnr    = raw[non_zero, 0].astype(int)
    maxop  = raw[non_zero, 2].astype(float)

    # path_f[j, i]: fractional count of path j contributed to ensemble i
    # path_w[j, i]: associated weight
    path_f = raw[non_zero, 4          : 3 + M].astype(float)
    path_w = raw[non_zero, 4 + M      : 3 + 2 * M].astype(float)

    # Per-path, per-ensemble weight ratio
    w = np.where(path_w != 0, path_f / path_w, 0.0)
    # Rescale so sum over paths equals the fractional sample count per ensemble
    col_sum = np.sum(w, axis=0)
    frac_sum = np.sum(path_f, axis=0)
    scale = np.where(col_sum != 0, frac_sum / col_sum, 0.0)
    w = w * scale                   # shape (N_paths, M)

    # Ensemble sizes (weighted)
    wsum = np.sum(w, axis=0)        # shape (M,)

    # WHAM crossing probability at TIS interfaces  P(lambda_i | lambda_0)
    N_ens = w.shape[1]   # actual number of ensemble columns in data (= M-1 typically)
    ploc = np.ones(N_ens + 1)   # P(lambda_i|lambda_0), i=0..N_ens. Gets later changed to 0 for i>0 if no paths cross that interface.
    for i in range(1, N_ens + 1):
        crosses_i = maxop >= interfaces[i]                   # shape (N_paths,)
        num = np.sum(crosses_i[:, None] * w[:, :i])
        den = np.sum(wsum[:i] / ploc[:i])
        ploc[i] = num / den if den > 0 else 0.0

    # Q_k = 1 / cumsum_{j=0}^{k} n_j / P(lambda_j | lambda_0)
    # wsum has shape (N_ens,), ploc has shape (N_ens+1,).
    # use ploc[:N_ens] (indices 0..N_ens-1) paired with wsum.
    cumden = np.cumsum(wsum / ploc[:N_ens])   # shape (N_ens,)
    Q = 1.0 / cumden                           # Q[k] for k = 0..N_ens-1

    # Per-trajectory unbiased weights  A[j] = Q[K(maxop[j])] * sum_i w[j,i]
    # K(lambda) = index of the highest TIS interface that is <= lambda,
    #             clamped to M-2 (the last "useful" ensemble).
    K_per_path = np.searchsorted(interfaces, maxop, side="right") - 1
    K_per_path = np.clip(K_per_path, 0, N_ens - 1)

    path_weights = Q[K_per_path] * np.sum(w, axis=1)   # shape (N_paths,)

    np.savetxt(
        out,
        np.column_stack([pnr, maxop, path_weights]),
        header="path_nr\tmax_op\tweight",
        fmt=["%8d", "%9.5f", "%16.8e"],
    )
    print(f"Per-trajectory weights saved to {out}.")

    # Fine-grid crossing probability  v(alpha) = P(lambda^alpha | lambda_0)
    # Algorithm from paper step 1-3.
    lam_min = interfaces[0]
    lam_max = np.max(maxop)
    subgrid = np.linspace(lam_min, lam_max, n_subinterfaces)

    v = np.zeros(n_subinterfaces)

    for j in range(len(maxop)):
        lmax_j = maxop[j]
        # K for this trajectory (same as above)
        k_j = K_per_path[j]
        Q_j = Q[k_j]
        w_total_j = np.sum(w[j])

        # For each alpha where interfaces[i_j] <= lambda^alpha < lmax_j
        # (the paper increments for each contributing ensemble i, but with WHAM
        # the trajectory carries its own Q weight regardless of which ensemble i
        # it was sampled from -- the third step of the algorithm multiplies by
        # Q_{K(alpha)} after summing raw counts; here we do it per-trajectory.)
        mask = (subgrid >= interfaces[k_j]) & (subgrid < lmax_j)
        v[mask] += w_total_j * Q_j

    # Note: after the loop v(alpha) already has Q_{K(alpha)} folded in because
    # we used Q[K_per_path[j]] per trajectory and only incremented for alpha in
    # [lambda_{k_j}, lambda_max(X)).  Trajectories from ensemble i only reach
    # up to their own interface range, so K(alpha) = K_per_path[j] for all
    # alpha they contribute to -- consistent with step 3 of the algorithm.

    
    np.savetxt(
        outP,
        np.column_stack([subgrid, v]),
        header="lambda\tP(lambda|lambda_0)",
        fmt=["%12.6f", "%16.8e"],
    )
    print(f"Crossing probability saved to {outP}.")

    path_data = {"pnr": pnr, "maxop": maxop, "weights": path_weights}
    return subgrid, v, path_data


def compute_pdf_matrices(
    toml,
    data,
    cv_values,
    subgrid,
    v,
    nskip,
    n_bins,
    out_r,
    out_u,
    overw,
):
    """Compute M_r(q, alpha, beta) and M_u(q, alpha, beta).

    For each trajectory X:
        - alpha : index of the sub-interface lambda^alpha that the trajectory
                  crosses (lambda^alpha < lambda_max(X))
        - bin q : bin index of CV(x^{lambda^alpha}) -- the CV value at the
                  first crossing of lambda^alpha
        - beta  : sub-interface index for lambda^r (the "reactive" target)

    M_r[q, alpha, beta] accumulates Q_{K(lambda_max)} when lambda_max >= lambda^beta
    M_u[q, alpha, beta] accumulates Q_{K(lambda_max)} when lambda_max <  lambda^beta

    After accumulation, both matrices are normalised by v(alpha)
    (= P(lambda^alpha | lambda_0)) to convert to probability densities.

    Algorithm from paper "Analyzing Complex Reaction Mechanisms Using Path Sampling":
      1. Set M_r = M_u = 0.
      2. For each trajectory X in ensemble i:
           a. Determine lambda_max and Q_{K(lambda_max)}.
           b. For each alpha where lambda^alpha < lambda_max:
                i.  Determine x^{lambda^alpha} and bin q.
                ii. For each beta where lambda^beta >  lambda_max:
                      M_u[q, alpha, beta] += Q_{K(lambda_max)}
                    For each beta where lambda^beta <= lambda_max:
                      M_r[q, alpha, beta] += Q_{K(lambda_max)}
      3. Normalise: M_r[q,a,b] /= v(alpha),  M_u[q,a,b] /= v(alpha).

    Parameters
    ----------
    toml       : infretis config (for interfaces)
    data       : infretis data file
    cv_values  : (N_paths,) array of CV values at each trajectory's first
                 crossing of *each* sub-interface alpha.  In practice this is
                 a 2-D array (N_paths, n_subinterfaces) where entry [j, alpha]
                 is the CV of path j at its first crossing of subgrid[alpha],
                 or NaN if the path doesn't reach that interface.
    subgrid    : (n_alpha,) fine lambda grid  (from compute_crossing_probability)
    v          : (n_alpha,) P(lambda^alpha | lambda_0)  (from above)
    nskip      : rows to skip
    n_bins     : number of CV bins q
    out_r/out_u: optional paths to save M_r, M_u as compressed .npz
    overw      : overwrite guard

    Returns
    -------
    bins    : (n_bins+1,) bin edges
    M_r     : (n_bins, n_alpha, n_alpha) normalised reactive   distribution
    M_u     : (n_bins, n_alpha, n_alpha) normalised unreactive distribution
    """
    _check_overwrite(out_r, overw)
    _check_overwrite(out_u, overw)

    with open(toml, "rb") as f:
        cfg = tomli.load(f)
    interfaces = np.asarray(cfg["simulation"]["interfaces"], dtype=float)
    M_ens = len(interfaces)

    raw = np.loadtxt(data, dtype=str)
    raw = raw[nskip:]
    non_zero = raw[:, 3] != "----"
    raw[raw == "----"] = "0.0"

    maxop  = raw[non_zero, 2].astype(float)
    path_f = raw[non_zero, 4        : 3 + M_ens].astype(float)
    path_w = raw[non_zero, 4 + M_ens: 3 + 2 * M_ens].astype(float)

    w = np.where(path_w != 0, path_f / path_w, 0.0)
    col_sum = np.sum(w, axis=0)
    frac_sum = np.sum(path_f, axis=0)
    scale = np.where(col_sum != 0, frac_sum / col_sum, 0.0)
    w = w * scale
    wsum = np.sum(w, axis=0)

    # Rebuild ploc and Q (same as in compute_crossing_probability)
    N_ens = w.shape[1]
    ploc = np.ones(N_ens + 1)
    for i in range(1, N_ens + 1):
        crosses_i = maxop >= interfaces[i]
        num = np.sum(crosses_i[:, None] * w[:, :i])
        den = np.sum(wsum[:i] / ploc[:i])
        ploc[i] = num / den if den > 0 else 0.0

    cumden = np.cumsum(wsum / ploc[:N_ens])
    Q = 1.0 / cumden

    K_per_path = np.searchsorted(interfaces, maxop, side="right") - 1
    K_per_path = np.clip(K_per_path, 0, N_ens - 1)

    n_alpha = len(subgrid)

    # CV range for binning (computed from finite values in cv_values)
    cv_finite = cv_values[np.isfinite(cv_values)]
    cv_min, cv_max = cv_finite.min(), cv_finite.max()
    bins = np.linspace(cv_min, cv_max, n_bins + 1)

    # Allocate matrices  (n_bins, n_alpha, n_alpha)
    M_r = np.zeros((n_bins, n_alpha, n_alpha))
    M_u = np.zeros((n_bins, n_alpha, n_alpha))

    # Main loop (algorithm step 2)
    for j in range(len(maxop)):
        lmax_j = maxop[j]
        k_j    = K_per_path[j]
        Q_j    = Q[k_j]

        # Sub-interfaces that this trajectory crosses: lambda^alpha < lmax_j
        alpha_mask = subgrid < lmax_j              # shape (n_alpha,)
        alpha_indices = np.where(alpha_mask)[0]

        if len(alpha_indices) == 0:
            continue

        for alpha in alpha_indices:
            # CV of trajectory j at its first crossing of subgrid[alpha]
            cv_j_alpha = cv_values[j, alpha]
            if not np.isfinite(cv_j_alpha):
                continue

            # Bin index q
            q = np.searchsorted(bins, cv_j_alpha, side="right") - 1
            q = int(np.clip(q, 0, n_bins - 1))

            # Beta indices where lambda^beta > lmax_j  -> unreactive
            beta_u = subgrid > lmax_j
            M_u[q, alpha, beta_u] += Q_j

            # Beta indices where lambda^beta <= lmax_j -> reactive
            beta_r = ~beta_u
            M_r[q, alpha, beta_r] += Q_j

    # Step 3: normalise by v(alpha)
    for alpha in range(n_alpha):
        if v[alpha] > 0:
            M_r[:, alpha, :] /= v[alpha]
            M_u[:, alpha, :] /= v[alpha]

    if out_r:
        np.savez_compressed(out_r, M_r=M_r, bins=bins, subgrid=subgrid)
        print(f"M_r saved to {out_r}.")
    if out_u:
        np.savez_compressed(out_u, M_u=M_u, bins=bins, subgrid=subgrid)
        print(f"M_u saved to {out_u}.")

    return bins, M_r, M_u


def compute_overlap_binned(
    M_r,
    M_u,
    bins,
    alpha,
    beta,
    P_lambda_r_given_lambda_c,
):
    """Binned overlap S and predictive power T for given (lambda^c, lambda^r).

    From the paper:
        S = (V_bin / P(lambda^r | lambda^c)) * sum_q r_q * u_q / (r_q + u_q)
        T = 1 - S

    where r_q = M_r[q, alpha, beta] and u_q = M_u[q, alpha, beta].

    Parameters
    ----------
    M_r, M_u   : distribution matrices from compute_pdf_matrices
    bins        : CV bin edges
    alpha, beta : sub-interface indices for lambda^c and lambda^r
    P_lambda_r_given_lambda_c : P(lambda^r | lambda^c)  (from crossing prob.)

    Returns
    -------
    S : overlap integral (binned estimate)
    T : predictive power = 1 - S
    """
    r = M_r[:, alpha, beta]
    u = M_u[:, alpha, beta]
    V_bin = (bins[-1] - bins[0]) / len(bins[:-1])

    denom = r + u
    safe = denom > 0
    S = (V_bin / P_lambda_r_given_lambda_c) * np.sum(r[safe] * u[safe] / denom[safe])
    T = 1.0 - S
    return float(S), float(T)


def compute_overlap_smooth(
    M_r,
    M_u,
    bins,
    alpha,
    beta,
    P_lambda_r_given_lambda_c,
):
    """Smooth (SG-filtered) overlap S and predictive power T.

    Uses the Savitzky-Golay approach described at the end of the paper:
      1. Order data points by CV, assign each to r or u category.
      2. Compute integrated distributions R(psi) and U(psi) using cumulative
         WHAM weights.
      3. Interpolate onto a fine regular grid.
      4. Extend with plateau boundaries (1/4 range on each side).
      5. Apply SG filter (2nd order, window ~1/16 of CV range).
      6. Differentiate to recover r and u densities.
      7. Compute S from the smooth densities.

    Parameters
    ----------
    M_r, M_u   : distribution matrices
    bins        : CV bin edges
    alpha, beta : sub-interface indices
    P_lambda_r_given_lambda_c : P(lambda^r | lambda^c)
    path_weights: optional per-trajectory WHAM weights (same as A from step I);
                  if None, uniform weights are assumed within each bin.

    Returns
    -------
    S        : smooth overlap integral
    T        : 1 - S
    cv_grid  : CV axis of the smooth estimate
    r_smooth : smooth r(psi) density
    u_smooth : smooth u(psi) density
    """
    r_bin = M_r[:, alpha, beta]  # (n_bins,)
    u_bin = M_u[:, alpha, beta]  # (n_bins,)
    n_bins = len(bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Total weight at each bin position (r + u)
    w_total = r_bin + u_bin

    # Cumulative integrals R and U on the bin-center grid
    W = np.sum(w_total)
    if W == 0:
        return 0.0, 1.0, bin_centers, r_bin, u_bin

    R_cum = np.cumsum(r_bin) / W
    U_cum = np.cumsum(u_bin) / W

    # Fine grid for interpolation
    n_fine = max(4 * n_bins, 200)
    cv_range = bins[-1] - bins[0]
    cv_fine = np.linspace(bins[0], bins[-1], n_fine)

    R_fine = np.interp(cv_fine, bin_centers, R_cum)
    U_fine = np.interp(cv_fine, bin_centers, U_cum)

    # Extend with plateaus: 1/4 of range on each side
    n_ext = n_fine // 4
    ext_left  = np.zeros(n_ext)
    ext_right_R = np.full(n_ext, R_fine[-1])
    ext_right_U = np.full(n_ext, U_fine[-1])

    R_ext = np.concatenate([ext_left, R_fine, ext_right_R])
    U_ext = np.concatenate([ext_left, U_fine, ext_right_U])

    # SG filter: 2nd order polynomial, window ~1/16 of CV range (rounded to odd)
    window_frac = max(1, int(round(n_fine / 16)))
    window_len = window_frac if window_frac % 2 == 1 else window_frac + 1
    window_len = max(window_len, 5)   # at least 5 points

    R_sg = savgol_filter(R_ext, window_length=window_len, polyorder=2)
    U_sg = savgol_filter(U_ext, window_length=window_len, polyorder=2)

    # Derivative -> recover densities; strip extensions
    dv = cv_range / (n_fine - 1)   # fine-grid spacing
    r_all = np.gradient(R_sg, dv)
    u_all = np.gradient(U_sg, dv)

    r_smooth = r_all[n_ext: n_ext + n_fine]
    u_smooth = u_all[n_ext: n_ext + n_fine]

    # Clamp negative values (artefacts of differentiation at boundaries)
    r_smooth = np.maximum(r_smooth, 0.0)
    u_smooth = np.maximum(u_smooth, 0.0)

    # Overlap integral  S = (1/P) * integral r(psi)*u(psi) / t(psi) dpsi
    t_smooth = r_smooth + u_smooth
    safe = t_smooth > 0
    integrand = np.where(safe, r_smooth * u_smooth / t_smooth, 0.0)
    S = np.trapezoid(integrand, cv_fine) / P_lambda_r_given_lambda_c
    T = 1.0 - S

    return float(S), float(T), cv_fine, r_smooth, u_smooth


def extract_cv_crossings(
    cv_dir,
    pnr_expected,
    subgrid,
    op_col,
    cv_cols,
    encoding,
    ):
    """Load per-trajectory .txt files and extract (averaged) crossing CV values.

    Parameters
    ----------
    cv_dir       : directory containing 1.txt, 2.txt, ... trajectory files.
    pnr_expected : 1-D int array of path numbers, in order (from infretis data).
    subgrid      : fine lambda grid, shape (N_sub,).
    op_col       : name of the order-parameter column (default ``OP_Lamb``).
    cv_cols      : CV column names to extract; None -> all except op_col.
    encoding     : text encoding of the .txt files.

    Returns
    -------
    cv_array   : (N_paths, N_cvs, N_sub) float64, NaN where not reached.
    cv_names   : list of CV column names, length N_cvs.
    pnr_loaded : copy of pnr_expected (rows for missing files are all-NaN).
    """
    cv_dir = Path(cv_dir)
    N_paths = len(pnr_expected)
    N_sub   = len(subgrid)

    # Discover column layout from first available file
    cv_names, op_idx, cv_idxs = _discover_columns(
        cv_dir, pnr_expected, op_col, cv_cols, encoding
    )
    N_cvs = len(cv_names)

    cv_array = np.full((N_paths, N_cvs, N_sub), np.nan, dtype=np.float64)

    missing = 0
    for row_j, pnr in enumerate(pnr_expected):
        fpath = cv_dir / f"{pnr}.txt"
        if not fpath.exists():
            missing += 1
            continue

        try:
            frames = _load_trajectory(fpath, encoding)
        except Exception as exc:
            warnings.warn(f"Could not parse {fpath}: {exc}", stacklevel=2)
            continue

        max_col = max(op_idx, max(cv_idxs, default=-1))
        if frames.shape[1] <= max_col:
            warnings.warn(
                f"{fpath}: expected >= {max_col + 1} columns, "
                f"got {frames.shape[1]} – skipping.",
                stacklevel=2,
            )
            continue

        op_vals = frames[:, op_idx]   # shape (T,)

        for alpha in range(N_sub):
            lam_alpha = subgrid[alpha]

            # Step 1: find first crossing frame
            first_idx = _first_crossing_idx(op_vals, lam_alpha)
            if first_idx is None:
                continue   # NaN already in array
            
            for k, ci in enumerate(cv_idxs):
                cv_array[row_j, k, alpha] = frames[first_idx, ci]

    if missing:
        warnings.warn(
            f"{missing}/{N_paths} trajectory files not found in {cv_dir}. "
            "Corresponding rows are all-NaN.",
            stacklevel=2,
        )

    return cv_array, cv_names, pnr_expected.copy()

# Helper functions for column discovery and trajectory loading
def _discover_columns(
    cv_dir,
    pnr_expected,
    op_col,
    cv_cols,
    encoding,
):
    """Return (cv_names, op_col_index, cv_col_indices) from the first
    available file's column-name line.

    File layout (4 header lines before the data):
      1: "# reactive" / "# non-reactive"
      2: "# <ensemble info>"
      3: "# <duplicate/edit info>"
      4: column names (no leading "#")
    """
    header_line: str | None = None
    for pnr in pnr_expected:
        fpath = cv_dir / f"{pnr}.txt"
        if fpath.exists():
            with fpath.open(encoding=encoding) as f:
                lines = [f.readline() for _ in range(4)]
            line = lines[3].strip()
            if line:
                header_line = line
                break

    if header_line is None:
        raise FileNotFoundError(
            f"No trajectory .txt files found in {cv_dir} for the given path numbers."
        )

    all_cols = header_line.split()

    if op_col not in all_cols:
        raise ValueError(
            f"Order-parameter column '{op_col}' not found in header.\n"
            f"Available columns: {all_cols}"
        )
    op_idx = all_cols.index(op_col)

    if cv_cols is None:
        cv_cols = [c for c in all_cols if c != op_col]
    else:
        missing_cols = [c for c in cv_cols if c not in all_cols]
        if missing_cols:
            raise ValueError(
                f"Requested CV columns not found in header: {missing_cols}\n"
                f"Available: {all_cols}"
            )

    cv_idxs = [all_cols.index(c) for c in cv_cols]
    return cv_cols, op_idx, cv_idxs


def _load_trajectory(fpath, encoding):
    """Load a trajectory file (skip header + comment lines).
    Returns float64 array of shape (T, N_cols)."""
    data = np.loadtxt(
        fpath, dtype=float, comments="#", skiprows=4, encoding=encoding
    )
    if data.ndim == 1:
        data = data[np.newaxis, :]   # single-frame trajectory edge case
    return data


def _first_crossing_idx(op_vals, threshold):
    """Index of the first frame where op_vals >= threshold, or None."""
    meets = op_vals >= threshold
    if not np.any(meets):
        return None
    return int(np.argmax(meets))   # argmax on bool gives first True index

# -------------------------------------------------------------------------------
def compute_2d_grids(
    M_r,
    M_u,
    bins,
    v,
    subgrid,
):
    """Compute T and S over the full (lambda^c, lambda^r) grid.

    For each valid pair (alpha, beta) with alpha < beta and v[alpha] > 0
    and v[beta] > 0, compute T_binned, S_binned, T_smooth, S_smooth, and
    P(lambda^r | lambda^c).  Pairs that don't satisfy these conditions are
    left as NaN.

    Parameters
    ----------
    M_r, M_u : (n_bins, n_alpha, n_alpha) from compute_pdf_matrices
    bins     : (n_bins+1,) CV bin edges
    v        : (n_alpha,) crossing probability
    subgrid  : (n_alpha,) lambda values on the fine grid

    Returns
    -------
    dict with 2D arrays of shape (n_alpha, n_alpha), each indexed [alpha, beta]:
        T_binned, S_binned, T_smooth, S_smooth, P_r_given_c
    All entries where alpha >= beta or v == 0 are NaN (upper triangle + diagonal).
    """
    n_alpha = len(subgrid)
    nan2d   = lambda: np.full((n_alpha, n_alpha), np.nan)

    T_bin_grid = nan2d()
    S_bin_grid = nan2d()
    T_sm_grid  = nan2d()
    S_sm_grid  = nan2d()
    P_grid     = nan2d()

    for alpha in range(n_alpha):
        if v[alpha] <= 0:
            continue
        for beta in range(alpha + 1, n_alpha):   # enforce alpha < beta
            if v[beta] <= 0:
                continue

            P_r_given_c = v[beta] / v[alpha]
            P_grid[alpha, beta] = P_r_given_c

            # Binned
            S_b, T_b = compute_overlap_binned(
                M_r, M_u, bins, alpha, beta, P_r_given_c
            )
            T_bin_grid[alpha, beta] = T_b
            S_bin_grid[alpha, beta] = S_b

            # Smooth
            S_s, T_s, *_ = compute_overlap_smooth(
                M_r, M_u, bins, alpha, beta, P_r_given_c
            )
            T_sm_grid[alpha, beta] = T_s
            S_sm_grid[alpha, beta] = S_s

    return {
        "T_binned":   T_bin_grid,
        "S_binned":   S_bin_grid,
        "T_smooth":   T_sm_grid,
        "S_smooth":   S_sm_grid,
        "P_r_given_c": P_grid,
    }


def plot_2d_grids(
    grids,
    subgrid,
    cv_name,
    out_path,
):
    """Save a figure with five subplots for one CV.

    Layout (1 row × 5 columns):
        T_binned | T_smooth | S_binned | S_smooth | P(λ^r|λ^c)

    x-axis : lambda^c  (alpha, rows of the grid)
    y-axis : lambda^r  (beta,  cols of the grid)

    Only the lower triangle (beta > alpha) is populated; the rest is masked.

    Parameters
    ----------
    grids    : output of compute_2d_grids
    subgrid  : fine lambda grid
    cv_name  : used for the figure title and file name
    out_path : path to save the .png file
    """
    panels = [
        ("T_binned",   "T binned",          "RdYlGn", 0.0, 1.0),
        ("T_smooth",   "T smooth",           "RdYlGn", 0.0, 1.0),
        ("S_binned",   "S binned",          "RdYlGn_r", 0.0, 1.0),
        ("S_smooth",   "S smooth",          "RdYlGn_r", 0.0, 1.0),
        ("P_r_given_c","P(λʳ|λᶜ)",         "Blues",   0.0, 1.0),
    ]

    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(4.5 * len(panels), 4.5),
        constrained_layout=True,
    )
    fig.suptitle(f"CV: {cv_name}", fontsize=13, fontweight="bold")

    lam = subgrid          # short alias
    extent = [lam[0], lam[-1], lam[-1], lam[0]]  # [xmin,xmax,ymax,ymin] for imshow

    for ax, (key, label, cmap, vmin, vmax) in zip(axes, panels):
        data = grids[key]

        # Mask NaN so imshow shows them in a distinct colour
        masked = np.ma.masked_invalid(data)

        im = ax.imshow(
            masked.T,               # transpose: rows=alpha(x), cols=beta(y) -> imshow rows=y
            origin="upper",         # lambda^r=0 at top (matches convention beta > alpha)
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(labelsize=7)

        ax.set_xlabel("λᶜ  (crossing surface)", fontsize=9)
        ax.set_ylabel("λʳ  (reactive target)",  fontsize=9)
        ax.set_title(label, fontsize=10)

        # Draw the diagonal (alpha == beta line) to separate valid / invalid region
        ax.plot([lam[0], lam[-1]], [lam[0], lam[-1]],
                color="black", lw=0.8, ls="--", alpha=0.6)

        ax.tick_params(axis="both", labelsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # Grey out the invalid region (alpha >= beta, i.e. above the diagonal)
        ax.fill_between(
            [lam[0], lam[-1]], [lam[0], lam[-1]], [lam[-1], lam[-1]],
            color="lightgrey", alpha=0.4, zorder=0,
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def PPA(
    toml: Annotated[ str, typer.Option("-toml", help="The .toml file") ] = "infretis.toml",
    data: Annotated[ str, typer.Option("-data", help="The infretis_data.txt file") ] = "infretis_data.txt",
    out_w: Annotated[ str, typer.Option("-outw", help="Output .txt of the path weights") ] = "path_weights.txt",
    out_mr: Annotated[ str, typer.Option("-outmr", help="Output .txt of the Mr matrix") ] = "mr.txt",
    out_mu: Annotated[ str, typer.Option("-outmu", help="Output .txt of the Mu matrix") ] = "mu.txt",
    cv_dir: Annotated[ str, typer.Option("-cv-dir", help="Path data folder with CV values in .txt files") ] = "ML",
    nskip: Annotated[ int, typer.Option( "-nskip", help="Skip the first nskip entries of the infretis_data.txt file",), ] = 1000,
    plot_dir: Annotated[str,typer.Option("-plot-dir", help="Directory to save per-CV heatmap figures")] = "ppa_plots",
    overw: Annotated[ bool, typer.Option("-O", help="Force overwriting of files") ] = False,
    n_subinterfaces: Annotated[ int, typer.Option("-n-subinterfaces", help="Number of fine-grid points spanning [lambda_0, lambda_max]") ] = 200,
    n_bins: Annotated[ int, typer.Option("-n_bins", help="Number of bins for the PDF matrices") ] = 50,
    ):
    #outP: Annotated[ str, typer.Option( "-outP", help="Write the binless WHAM crossing probability to outP"), ] = "", # written unconditionally
    #cv_column_1: Annotated[ str, typer.Option("-cv-col1", help="Column name of CV values") ] = "", #Not yet implemented
    #cv_column_2: Annotated[ str, typer.Option("-cv-col2", help="Column name of CV values") ] = "", #Not yet implemented
    """Run the complete predictive power analysis (PPA) for every CV.

    For each CV:
      1. Compute M_r / M_u over the full (lambda^c, lambda^r) grid.
      2. Compute T_binned, T_smooth, S_binned, S_smooth, P(lambda^r|lambda^c)
         for every valid (alpha, beta) pair with alpha < beta.
      3. Save a heatmap figure with one subplot per quantity.

    Returns
    -------
    dict:
      subgrid  - fine lambda grid
      v        - crossing probability on the grid
      results  - list of per-CV dicts sorted by mean T_smooth, each containing:
                   cv, bins, M_r, M_u, grids (the 2D arrays), plot_path
    """
    # Step I: crossing probability
    subgrid, v, path_data = compute_crossing_probability(
        toml, data, out_w, outP="p_data.txt",
        nskip=nskip, overw=overw, n_subinterfaces=n_subinterfaces,
    )

    positive_idx = np.where(v > 0)[0]
    if len(positive_idx) < 2:
        raise ValueError(
            "Fewer than 2 sub-interfaces have P > 0. "
            "Increase n_subinterfaces or check the data."
        )
    print(
        f"Crossing probability computed on {n_subinterfaces} sub-interfaces; "
        f"{len(positive_idx)} have P > 0."
    )

    print(f"  Subgrid: [{subgrid[0]:.4f}, {subgrid[-1]:.4f}], {n_subinterfaces} points")

    # load CV crossing values
    raw = np.loadtxt(data, dtype=str)
    raw = raw[nskip:]
    non_zero = raw[:, 3] != "----"
    pnr = raw[non_zero, 0].astype(int)

    cv_array, cv_names, _ = extract_cv_crossings(
        cv_dir=cv_dir,
        pnr_expected=pnr,
        subgrid=subgrid,
        op_col="OP_Lamb",
        cv_cols=None,  # extract all CV columns except OP_Lamb
        encoding="utf-8",
    )   # returns tuple of: cv_array of shape (N_paths, N_cvs, n_subinterfaces)
        #                   cv_names list of CV column names
        #                   pnr_loaded array of path numbers (same order as cv_array)
    N_cvs = len(cv_names)
    print(f"{len(pnr)} paths - {N_cvs} CVs - {len(subgrid)} sub-interfaces.")

    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for k, cv_name in enumerate(cv_names):
        print(f"\n[{k+1:3d}/{N_cvs}] {cv_name}")

        cv_values_k = cv_array[:, k, :]   # (N_paths, N_sub)

        # Skip CVs with too little data at any positive-P interface
        n_valid = np.array([
            np.sum(np.isfinite(cv_values_k[:, a])) for a in positive_idx
        ])
        if np.any(n_valid < 5):
            print(f"  SKIP: fewer than 5 valid crossings at "
                  f"{np.sum(n_valid < 5)} sub-interface(s).")
            results.append({"cv": cv_name, "grids": None,
                            "mean_T_smooth": np.nan})
            continue

        bins, M_r, M_u = compute_pdf_matrices(
            toml, data, cv_values_k, subgrid, v,
            nskip=nskip, n_bins=n_bins,
            out_r=str(Path(out_mr).with_stem(f"{Path(out_mr).stem}_{cv_name}")),
            out_u=str(Path(out_mu).with_stem(f"{Path(out_mu).stem}_{cv_name}")),
            overw=overw,
        )

        # --- 2D grid over all (alpha, beta) pairs ---
        grids = compute_2d_grids(M_r, M_u, bins, v, subgrid)

        mean_T_smooth = float(np.nanmean(grids["T_smooth"]))
        mean_T_binned = float(np.nanmean(grids["T_binned"]))
        print(f"  mean T_smooth = {mean_T_smooth:.4f}  "
              f"mean T_binned = {mean_T_binned:.4f}")

        # --- heatmap figure ---
        safe_name = cv_name.replace("/", "_").replace(" ", "_")
        plot_path = str(Path(plot_dir) / f"{safe_name}.png")
        plot_2d_grids(grids, subgrid, cv_name, plot_path)

        results.append({
            "cv":            cv_name,
            "bins":          bins,
            "M_r":           M_r,
            "M_u":           M_u,
            "grids":         grids,
            "mean_T_smooth": mean_T_smooth,
            "mean_T_binned": mean_T_binned,
            "plot_path":     plot_path,
        })

    results.sort(
        key=lambda d: d["mean_T_smooth"] if np.isfinite(d["mean_T_smooth"]) else -1.0,
        reverse=True,
    )

    print(f"\n{'Rank':>4}  {'CV':<30}  {'mean T_smooth':>14}  {'mean T_binned':>14}")
    print("-" * 68)
    for rank, r in enumerate(results, 1):
        T_sm  = f"{r['mean_T_smooth']:.4f}" if np.isfinite(r["mean_T_smooth"]) else "   NaN"
        T_bin = f"{r['mean_T_binned']:.4f}" if "mean_T_binned" in r and np.isfinite(r.get("mean_T_binned", np.nan)) else "   NaN"
        print(f"{rank:>4}  {r['cv']:<30}  {T_sm:>14}  {T_bin:>14}")

"""
app = typer.Typer()

@app.command()
def main(
    toml: Annotated[str,  typer.Option("-toml")]      = "infretis.toml",
    data: Annotated[str,  typer.Option("-data")]      = "infretis_data.txt",
    out_w: Annotated[str, typer.Option("-outw")]      = "path_weights.txt",
    out_mr: Annotated[str,typer.Option("-outmr")]     = "mr.npz",
    out_mu: Annotated[str,typer.Option("-outmu")]     = "mu.npz",
    cv_dir: Annotated[str,typer.Option("-cv-dir")]    = "ML",
    plot_dir: Annotated[str,typer.Option("-plot-dir")]= "ppa_plots",
    nskip: Annotated[int, typer.Option("-nskip")]     = 1000,
    overw: Annotated[bool,typer.Option("-O")]         = False,
    n_subinterfaces: Annotated[int, typer.Option("-n-subinterfaces")] = 200,
    n_bins: Annotated[int,typer.Option("-n_bins")]    = 50,
) -> None:
    PPA(
        toml=toml, data=data, out_w=out_w, out_mr=out_mr, out_mu=out_mu,
        cv_dir=cv_dir, plot_dir=plot_dir, nskip=nskip, overw=overw,
        n_subinterfaces=n_subinterfaces, n_bins=n_bins,
    )


if __name__ == "__main__":
    app()"""