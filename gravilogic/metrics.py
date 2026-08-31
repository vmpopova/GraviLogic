"""gravilogic/metrics.py — primitive and composite geometric audit metrics."""
# Three-layer structure:
#   Layer 1 — PRIMITIVE metrics: each measures one distinct source of
#             information about the trajectory (path geometry, local
#             model behavior, data support, per-feature attribution).
#   Layer 2 — COMPOSITE metrics: explicit interactions between primitives
#             from different sources, each built to answer one specific
#             diagnostic question.
#   Layer 3 — GraviLogic Score: a single number combining Layers 1+2,
#             defined in score.py (not this file) once weights are
#             calibrated — this file only produces the inputs to it.
#
# This module went through an empirical revision: it originally had a
# fourth primitive ("Length," total path distance) and treated all axes
# as mathematically independent by design. Real testing on HTRU2 (pulsar
# candidate dataset, 17898 rows, MLP classifier) showed neither holds:
# Length correlated with Bending Energy at Spearman r=0.95 (redundant,
# dropped), and the axes are NOT statistically independent on tabular
# data of this kind. The metrics below are kept anyway, because the goal
# is not mathematical orthogonality for its own sake — it's catching
# model instability, and correlated signals reinforcing each other on a
# genuinely unstable point is a feature of a good audit, not a flaw.
#
# EXPERIMENTAL RESULTS ARE TRACKED PER-FUNCTION BELOW AS COMMENTS, NOT
# folded into the theoretical description — so it's always clear which
# claims are "this is what the formula computes" (permanent, theoretical)
# versus "this is what we found when we tested it on HTRU2" (a dated,
# specific experimental result that could change on other data).

from __future__ import annotations
import numpy as np
from .core import Trajectory


# =============================================================================
# LAYER 1 — PRIMITIVE METRICS
# =============================================================================

# ---- Source: geometry of the path itself (depends only on z_path) --------
#
# NOTE: an earlier version also had "Length" (L(gamma) = sum of step
# distances). DROPPED after testing — see module docstring. If you need
# raw path length for some other purpose, it's a one-line sum over
# z_path; just don't reintroduce it as a separate reported metric without
# re-checking its correlation with Bending Energy on your own data first.

def bending_energy(traj: Trajectory) -> np.ndarray:
    """
    Bending Energy B(gamma), per interior step (array, not pre-summed —
    composite metrics below need to know WHERE the max occurs).

    Definition, per interior point i:
        b_i = || z_{i+1} - 2*z_i + z_{i-1} ||_2^2
    Discrete second derivative of the curve, squared and normed — the
    discrete analogue of elastic bending energy (integral of curvature^2
    along a curve) from differential geometry / spline theory.

    # EXPERIMENT (HTRU2, MLP hidden=(16,), n=80 misclassified vs 80
    # matched-correct, Mann-Whitney U): mean B along the path is
    # significantly higher for misclassified points. AUC=0.724, p<0.0001.
    # Second-strongest single signal found so far, after manifold_support.
    """
    second_diff = traj.z_path[2:] - 2 * traj.z_path[1:-1] + traj.z_path[:-2]
    # f''(x) ≈ f(x+h) - 2f(x) + f(x-h), applied to consecutive trajectory
    # points. Interior points only — the first/last points have no
    # "both neighbors" to compute this from.
    return np.sum(second_diff ** 2, axis=-1)
    # Per-step squared norm — length T-2.


# ---- Source: local model behavior (depends on h, evaluated pointwise) ----

def local_sensitivity(jacobian_fn, x_path: np.ndarray) -> np.ndarray:
    """
    Local Sensitivity, ||J(t)||_F per point — the local Lipschitz constant
    of the encoder h at each interpolated input point.

    jacobian_fn: x -> ||J||_F (scalar). Architecture-specific: closed-form
    for a ReLU-MLP, autograd for anything else, a Hutchinson trace
    estimator for cheap approximation at scale (see the AXIOM draft for
    that trick applied to a transformer).

    # EXPERIMENT (HTRU2, same setup as above): mean sensitivity is LOWER
    # for misclassified points (3.30) than correct ones (4.11) —
    # AUC=0.175 (computed the normal "high=risky" way, so far below 0.5
    # that it is itself evidence of a real, strong, INVERTED relationship,
    # not noise; p<0.0001). Using max instead of mean along the path gives
    # the same inverted direction (AUC=0.204) — not an aggregation
    # artifact. Tentative interpretation: on this dataset the model may
    # be locally "flat and confidently wrong," not locally unstable, at
    # points it misclassifies — the opposite failure mode this metric was
    # originally designed to catch. NOT included in the current audit
    # rule (see score.py) until the sign/interpretation is resolved.
    """
    return np.array([jacobian_fn(x_path[i]) for i in range(x_path.shape[0])])


def output_smoothness(traj: Trajectory) -> np.ndarray:
    """
    Output Smoothness, per interior step: u_i = |f_{i+1} - 2f_i + f_{i-1}|.
    Discrete second derivative of the scalar model OUTPUT along the path
    — the one metric here that looks at f, not z.

    # EXPERIMENT: not yet tested against misclassification. Defined and
    # mathematically distinct from bending_energy (a straight path in z
    # can still have sharply-changing f, and vice versa), but its
    # practical value is an open question, not a validated claim.
    """
    return np.abs(traj.f_path[2:] - 2 * traj.f_path[1:-1] + traj.f_path[:-2])


# ---- Source: data support (statistical, independent of path/model) -------

def manifold_support(density_fn, path: np.ndarray) -> np.ndarray:
    """
    Manifold Support, rho(point) per point — KDE-based density estimate of
    how "typical" each point is relative to real training data.

    path: x_path or z_path, matching whatever space density_fn was fit on.

    # EXPERIMENT: the strongest validated signal in this module. Minimum
    # density along the path is ~20x lower for misclassified points
    # (mean 0.008) than correct ones (mean 0.154). AUC=0.793, p<0.0001,
    # n=80/80 — the best single-metric AUC found. Anchor metric of the
    # current audit rule (see score.py): every tested flag combination
    # that included this metric outperformed every combination that
    # didn't.
    """
    # ARCHITECTURE-DEPENDENT IMPLEMENTATION NOTE: KDE-based density_fn
    # (as used for the HTRU2 experiments, 8 input dimensions) breaks down
    # in high-dimensional embedding spaces — tested directly on RoBERTa-
    # base's 768-dim hidden states, where scipy's gaussian_kde either
    # fails outright (full-dim: singular covariance matrix) or returns
    # near-constant, uninformative values (after PCA to 30 dims,
    # explained variance 0.949: density range ratio 1.15x — no real
    # signal). k-NN distance to the k nearest neighbors in the same
    # embedding space (NOT a density function, but the same intent — "how
    # typical/supported is this point") was tested as a drop-in
    # replacement and validated: AUC=0.777 (p<0.0001) for detecting
    # misclassification on SST-2/RoBERTa-base, n=80/80, stable across
    # k=3..15. Use KDE for low-dimensional inputs (roughly <20 dims,
    # confirmed working on HTRU2's 8 dims); use k-NN distance for
    # high-dimensional embeddings (confirmed on 768 dims). Mahalanobis
    # distance was also tested and FAILED (AUC=0.489, degenerate
    # covariance estimate with n=80 << 768 dimensions) — not recommended
    # without a shrinkage estimator (e.g. Ledoit-Wolf) at minimum, and
    # untested even then.
    """
    return np.array([density_fn(path[i]) for i in range(path.shape[0])])


# ---- Source: per-feature attribution --------------------------------------

def trajectory_deviation_mass(traj: Trajectory) -> float:
    """
    Trajectory Deviation Mass (TDM) — renamed from an earlier "Cognitive
    Mass" / "Path-Integrated Attribution" specifically to avoid confusion
    with the established Integrated Gradients family (Sundararajan et al.
    2017), which computes something related but formally different (a
    path integral of the GRADIENT, not a variance-weighted sum of
    deviations from the path's own mean).

    Definition: TDM(gamma) = sum_i || z_i - z_bar ||_2^2 * | f_i - f_bar |
    Discrete analogue of a physics work integral — accumulated internal
    representation drift, weighted by output deviation, over the WHOLE
    path, rather than a single before/after comparison (what SHAP/LIME
    compute).

    # EXPERIMENT: confirmed. Mean TDM significantly higher for
    # misclassified points (3.98) than correct (2.97), AUC=0.684,
    # p=0.0001, n=80/80. On an earlier, smaller sample (n=60/60) this
    # looked like noise (AUC=0.521) — the larger sample had the
    # statistical power the smaller one lacked. Included in the current
    # audit rule.

    To get a PER-FEATURE value: call this once per feature j, using a
    trajectory built from a counterfactual x' constrained (via
    counterfactuals.py's allowed_features=[j]) to perturb ONLY feature j.
    """
    z_bar = traj.z_path.mean(axis=0, keepdims=True)
    f_bar = traj.f_path.mean()
    sq_dev = np.sum((traj.z_path - z_bar) ** 2, axis=-1)
    return float(np.sum(sq_dev * np.abs(traj.f_path - f_bar)))


# =============================================================================
# LAYER 2 — COMPOSITE METRICS
# =============================================================================
# Each combines two primitives from DIFFERENT sources above, to answer a
# question neither answers alone. Every function takes already-computed
# primitive arrays as input (never recomputes them), so the dependency is
# explicit and no composite can accidentally combine two things from the
# same source under a new name.

def orbit_decay(bending: np.ndarray, sensitivity: np.ndarray) -> float:
    """
    Orbit Decay = max_t [ bending(t) * sensitivity(t) ] — "is the path
    bending AND is the model sensitive, at the same point?" Multiplication
    is deliberate: if either factor is near zero at a point, the product
    is near zero there too, regardless of the other — only the
    CONJUNCTION of both triggers a high value.

    bending, sensitivity must be pre-aligned to the same trajectory
    indices first — see align_to_interior() below.

    # EXPERIMENT: confirmed individually (AUC=0.716, p<0.0001), but found
    # LARGELY REDUNDANT with Bending Energy once tested inside the audit
    # rule — adding it to a rule that already includes Bending Energy did
    # not improve F1 (0.650 vs 0.654 without it). Consistent with the
    # fact that Sensitivity's own signal runs in the opposite, confounding
    # direction here (see local_sensitivity). NOT part of the current
    # recommended audit rule for that reason — kept for datasets where
    # Sensitivity might behave differently.
    """
    return float(np.max(bending * sensitivity))


def unsupported_bending(bending: np.ndarray, density: np.ndarray, eps: float = 1e-8) -> float:
    """
    Unsupported Bending = max_t [ bending(t) / (density(t) + eps) ] —
    "is the path bending specifically where there's little data support?"
    Division (not multiplication) because the question is "one high, the
    OTHER low," not "both high together."

    # EXPERIMENT: not yet tested directly against misclassification —
    # a natural next step, since both ingredients (Bending Energy,
    # Manifold Support) are the two strongest validated primitives.
    """
    return float(np.max(bending / (density + eps)))


def unsupported_sensitivity(sensitivity: np.ndarray, density: np.ndarray, eps: float = 1e-8) -> float:
    """
    Unsupported Sensitivity = max_t [ sensitivity(t) / (density(t) + eps) ].

    # CAUTION: since Sensitivity alone runs INVERTED relative to
    # misclassification risk on HTRU2, this composite likely inherits
    # that inversion. Not tested directly — flagged here so it isn't
    # mistaken for a clean signal without checking first.
    """
    return float(np.max(sensitivity / (density + eps)))


def extrapolation_attribution(traj: Trajectory, density: np.ndarray, density_threshold: float) -> float:
    """
    Extrapolation Attribution — Trajectory Deviation Mass, restricted to
    only the points where density falls below threshold (TDM computed
    only over the "unsupported" part of the journey).

    # EXPERIMENT: not yet tested — a natural follow-up given that both
    # TDM and Manifold Support are independently validated strong
    # signals, but their combination hasn't been run yet.
    """
    z_bar = traj.z_path.mean(axis=0, keepdims=True)
    f_bar = traj.f_path.mean()
    sq_dev = np.sum((traj.z_path - z_bar) ** 2, axis=-1)
    raw_terms = sq_dev * np.abs(traj.f_path - f_bar)
    mask = density < density_threshold
    return float(np.sum(raw_terms[mask]))


# =============================================================================
# HELPER — aligning arrays of different lengths (T vs T-2) before combining
# =============================================================================

def align_to_interior(full_length_array: np.ndarray) -> np.ndarray:
    """
    bending_energy() and output_smoothness() return T-2 values (interior
    points only). local_sensitivity() and manifold_support() return T
    values (every point). Trim the full-length array to match before
    combining them in a composite metric.
    """
    return full_length_array[1:-1]
