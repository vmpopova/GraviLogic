"""gravilogic/core.py — core framework structures."""
# Module with the foundation of the GraviLogic method: the model
# decomposition f = g∘h, the trajectory γ(t) between the factual and
# counterfactual points, and a helper that ties the two together so
# counterfactuals.py and core.py agree on what a "prediction function"
# means. Only base data structures go here — the actual metrics (length,
# curvature, sensitivity, etc.) live in metrics.py, and the axiom-
# constrained search for x' lives in counterfactuals.py.

from dataclasses import dataclass
# dataclass — a Python decorator that auto-generates __init__ and other
# boilerplate methods for a data-container class, so we don't write them
# by hand.

import numpy as np
# numpy is needed for working with vectors/arrays (latent points, input
# data — all of it is numpy arrays).


class Encoder:
    """h: X -> Z, a wrapper around any model with an explicit latent layer."""
    # This is the code implementation of the function h from the theory
    # (f = g ∘ h). h turns an input x into a point in latent space
    # z = h(x). The class exists because "extracting the right layer's
    # activations" is done differently for different architectures
    # (MLP, XGBoost, transformer) — the wrapper hides that difference
    # behind one common interface.

    def __init__(self, model, extract_layer):
        # __init__ — the class constructor, called when creating an
        # object: Encoder(model, extract_layer).
        self.model = model
        # Store the trained model itself (e.g. your MLP or XGBoost)
        # inside the object, to use later in __call__.
        self.extract_layer = extract_layer
        # Store the strategy function: it knows how to extract the
        # latent representation specifically from THIS model.
        # E.g. for an MLP this might be "take the output of the
        # second-to-last layer"; for XGBoost it would be something
        # entirely different; for a transformer, hidden_states pooled
        # over tokens. extract_layer — the function that pulls out the
        # target layer's activations.

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # __call__ — a special Python method: it lets an Encoder object
        # be called like a function, i.e. encoder(x) instead of
        # encoder.extract_layer(encoder.model, x).
        return self.extract_layer(self.model, x)
        # Pass the model and input x into the stored strategy function
        # and return the result — this is z = h(x).


@dataclass
class Trajectory:
    """γ(t): [0,1] -> Z, discretized into T points (Definition 1)."""
    # On paper, γ(t) is a continuous trajectory in latent space.
    # A computer can't compute continuity, so we store a set of
    # discrete points of that trajectory instead — that's Trajectory.

    x_path: np.ndarray   # shape (T, n) — interpolated INPUT points x(t_i)
    # Array of shape (T, n), n = input dimensionality. Each row is the
    # intermediate input x(t_i) = (1-t_i)x + t_i*x' BEFORE it goes through
    # the encoder. Stored here (not just derived ad hoc elsewhere) because
    # metrics.py's local_sensitivity() needs the Jacobian of h evaluated
    # AT these exact input points (the Jacobian of h is a function of x,
    # not of z), and manifold_support() can optionally be computed in
    # input space too. Keeping x_path as part of Trajectory means there is
    # only ONE place that defines the interpolation formula — metrics
    # can never silently drift out of sync with what actually produced
    # z_path and f_path below.

    z_path: np.ndarray   # shape (T, d) — latent points γ(t_i)
    # Array of shape (T, d): T is the number of discretization points,
    # d is the dimensionality of latent space Z. Each row z_path[i] is
    # one point γ(t_i), i.e. the result of passing x_path[i] through
    # the encoder.

    t: np.ndarray        # shape (T,)  — values of parameter t
    # Array of length T with values of parameter t from 0 to 1 —
    # i.e. "how far along the transition" each point is. t[0] = 0
    # corresponds to the original x, t[-1] = 1 to the counterfactual x'.

    f_path: np.ndarray   # shape (T,)  — predictions g(γ(t_i)) along the path
    # Array of length T: the model's prediction (output of g) at each
    # point of the trajectory. Needed for metrics that look at how the
    # prediction itself changes (e.g. output smoothness, Trajectory
    # Deviation Mass), not just the geometry of z.

    @property
    def T(self) -> int:
        # @property lets us access T as an attribute (trajectory.T)
        # rather than as a method (trajectory.T()) — purely for
        # readability.
        return len(self.t)
        # The number of discretization points is just the length of t.


def build_trajectory(
    encoder: Encoder,
    x: np.ndarray,
    x_prime: np.ndarray,
    predict_from_latent,
    T: int = 50,
) -> Trajectory:
    """γ(0)=h(x), γ(1)=h(x'), linear interpolation in input space (Eq. 9)."""
    # The function that actually BUILDS the trajectory for a specific
    # pair (x, x'). Takes:
    # encoder            — an Encoder object (i.e. h)
    # x                  — the original (factual) input
    # x_prime            — the counterfactual input x' (produced by
    #                       counterfactuals.select_counterfactual)
    # predict_from_latent — a function z -> prediction, i.e. THIS IS g,
    #                       the decoder half of f = g∘h. Named explicitly
    #                       "from_latent" (rather than a generic
    #                       "predict_fn") so it can never be confused
    #                       with the full end-to-end f(x) used in
    #                       counterfactuals.py — see
    #                       compose_predict_from_input() below, which
    #                       builds f(x) FROM this function, so you only
    #                       ever have to define g once, in one place.
    # T                  — how many discretization points to use.

    t = np.linspace(0.0, 1.0, T)
    # np.linspace(0, 1, T) creates T evenly spaced numbers from 0 to 1
    # inclusive — the discretization nodes for parameter t.

    x_path = np.array([(1 - ti) * x + ti * x_prime for ti in t])
    # For each ti in t, compute the point on the line between x and x':
    # (1 - ti) * x + ti * x'. At ti=0 we get x, at ti=1 we get x', in
    # between — the intermediate points. This is x(t) = (1-t)x + tx'.
    # [... for ti in t] is a list comprehension, a compact loop that
    # builds a list directly; np.array(...) turns it into a single
    # numpy array of shape (T, n).

    z_path = np.array([encoder(xi) for xi in x_path])
    # Pass EVERY point of the interpolated input through the encoder
    # (i.e. through h), getting the corresponding latent point
    # z = h(x(t)). Result: array of all trajectory points γ(t) in
    # latent space, shape (T, d).

    f_path = np.array([predict_from_latent(zi) for zi in z_path])
    # For each latent point zi, compute the model's prediction
    # predict_from_latent(zi) — i.e. apply g. Gives us how the model's
    # output changes along the whole trajectory.

    return Trajectory(x_path=x_path, z_path=z_path, t=t, f_path=f_path)
    # Bundle everything — including x_path, needed downstream by
    # metrics.py — into one Trajectory object and return it.


def compose_predict_from_input(encoder: Encoder, predict_from_latent):
    """
    Builds f(x) = g(h(x)) automatically from h (encoder) and g
    (predict_from_latent), so the FULL model function used by
    counterfactuals.py is always derived from — and therefore
    guaranteed consistent with — the exact same g used inside
    build_trajectory(). This removes an earlier inconsistency where
    core.py and counterfactuals.py each independently expected a
    differently-shaped "predict_fn" (one over z, one over x), with
    nothing stopping the two from silently disagreeing.
    """
    # WHY THIS FUNCTION EXISTS: without it, you'd have to write TWO
    # separate prediction functions by hand — one taking z (for
    # build_trajectory) and one taking x (for select_counterfactual) —
    # and nothing would guarantee they agree (different preprocessing,
    # a bug fixed in one but not the other). Deriving the x-based
    # function FROM the z-based one means there is only ONE place where
    # g is defined, and the two modules are consistent by construction.

    def predict_from_input(x: np.ndarray):
        # Takes a raw input x (not yet encoded) and returns f(x).
        return predict_from_latent(encoder(x))
        # encoder(x) computes z = h(x). predict_from_latent(z) computes
        # g(z). Composed: g(h(x)) = f(x), matching f = g∘h.

    return predict_from_input
    # Return the composed function itself — used as predict_from_input
    # in counterfactuals.select_counterfactual().

