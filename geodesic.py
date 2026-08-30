"""gravilogic/geodesic.py — optimal-path comparison via a Riemannian pullback metric."""
# WHY THIS MODULE EXISTS: every prior version of the method (the PDF
# draft, the AXIOM draft, the thesis) explicitly left one thing undone —
# comparing the model-induced trajectory γ (a straight-line interpolation
# in input space, pushed through h) against a theoretically OPTIMAL path
# γ* between the same two points. Without γ*, there was no way to say
# "how much does the model's actual behavior deviate from the best
# possible path" — only "what does the model's actual path look like."
# This module closes that gap.
#
# WHERE THIS COMES FROM: adapted from Pegios, Feragen, Hansen, Arvanitidis,
# "Counterfactual Explanations via Riemannian Latent Space Traversal"
# (arXiv:2411.02259, NeurIPS 2024 Workshop NeurReps). Their method builds
# a Riemannian metric in a generative model's latent space, pulled back
# from a metric defined using a CLASSIFIER's hidden representation, and
# finds geodesics (shortest paths under that metric) for counterfactual
# generation. GraviLogic does NOT need their generative-decoder part —
# we already have direct access to a classifier's encoder h, so we use
# only their classifier-side construction (their Eq. 4):
#
#     M_X(x) = J_h(x)^T J_h(x)
#
# — a Riemannian metric directly on input space X, built from the
# Jacobian of the SAME encoder h that GraviLogic already uses everywhere
# else (this is exactly local_sensitivity's Jacobian, reused here for a
# different purpose). This is the adaptation: their method pulls this
# metric further back into a VAE's latent space because they need to
# optimize IN that latent space; we skip that step entirely and work
# directly in X, because GraviLogic has no generative model and doesn't
# need one for this purpose.
#
# WHAT THIS BUYS US: under M_X, straight lines in Euclidean space are no
# longer geodesics — shortest paths bend AWAY from regions where the
# classifier's representation changes sharply (where M_X is "expensive"),
# similar in spirit to how Pegios et al.'s metric makes paths avoid
# data-sparse regions. Computing the actual geodesic γ* between x and x'
# under M_X, and comparing it to the plain-interpolation γ that
# build_trajectory() already produces, gives a genuinely new metric:
# how much does the model's real behavior deviate from the path a
# "well-behaved" Riemannian-optimal traversal would take?
#
# STATUS OF THIS FILE: architectural placeholder + the one function
# (the metric itself) that's a direct, faithful port of Pegios et al.'s
# published formula. The geodesic SOLVER (finding γ* by minimizing path
# length under M_X — their Riemannian SGD, or any equivalent numerical
# geodesic solver) is NOT implemented yet. Nothing here has been run or
# validated empirically — unlike metrics.py, which documents real HTRU2
# results, every claim in this file is theoretical / literature-derived
# until an actual experiment says otherwise. Do not treat anything below
# as a confirmed result.

from __future__ import annotations
import numpy as np


def pullback_metric(jacobian_matrix_fn, x: np.ndarray) -> np.ndarray:
    """
    M_X(x) = J_h(x)^T J_h(x) — Pegios et al. 2024, Eq. 4, ported directly.

    jacobian_matrix_fn: x -> full Jacobian matrix of h at x, shape (d, n)
    (d = latent dim, n = input dim) — NOT the scalar Frobenius norm used
    by metrics.local_sensitivity(); this needs the full matrix because
    the metric M_X is itself a matrix (n x n), used to define an inner
    product on tangent vectors in input space, not a single number.

    Returns: an (n, n) positive semi-definite matrix. Intuition from the
    source paper: M_X is small where the classifier's representation is
    locally stable (cheap to move through), and large near regions where
    small input changes cause large representation changes (expensive to
    move through) — so shortest paths under this metric are pushed away
    from those regions, similarly to how Pegios et al. describe their
    metric penalizing paths that cross decision-boundary-adjacent zones.
    """
    J = jacobian_matrix_fn(x)
    return J.T @ J
    # J^T @ J: standard construction of a PSD matrix from any matrix J —
    # this is exactly Pegios et al.'s Eq. 4, applied to our encoder's
    # Jacobian instead of theirs.


def geodesic_path(x, x_prime, jacobian_matrix_fn, T=50):
    """
    NOT YET IMPLEMENTED.

    Intended to return the geodesic gamma*(t) between x and x' under the
    metric pullback_metric(), analogous to Pegios et al.'s Riemannian SGD
    (their Eq. 3) but solved directly in input space X rather than in a
    VAE's latent space Z (since we have no VAE — see module docstring).

    A real implementation needs: (1) a numerical geodesic solver (e.g.
    minimizing the discretized Riemannian path-length functional via
    gradient descent on the sequence of intermediate points, analogous to
    Riemannian SGD but without a generative decoder in the loop), and
    (2) validation that the resulting path actually differs meaningfully
    from build_trajectory()'s straight-line interpolation on real data —
    neither of which has been done yet. Left unimplemented deliberately
    rather than filled in with an unvalidated guess.
    """
    raise NotImplementedError(
        "geodesic_path is a planned extension (see module docstring for "
        "the Pegios et al. adaptation this is based on) — not yet built "
        "or tested."
    )


def geodesic_deviation(length_gamma: float, length_gamma_star: float) -> float:
    """
    D(gamma, gamma*) = | length(gamma) - length(gamma*) |

    The simplest possible comparison between the model-induced trajectory
    and the optimal one: absolute difference in total path length (both
    measured under the SAME metric — this needs the Riemannian length of
    gamma too, not the Euclidean bending_energy/length from metrics.py,
    for a fair comparison — that computation isn't wired up yet either).

    STATUS: placeholder formula, chosen for simplicity, NOT validated.
    An alternative (comparing bending energy instead of length, or a
    genuine point-wise comparison after arc-length reparametrization)
    may turn out to be more informative — this is an open experimental
    question, not a settled design choice.
    """
    return abs(length_gamma - length_gamma_star)
