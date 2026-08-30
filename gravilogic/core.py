"""gravilogic/core.py — core framework structures ."""
# Module with the foundation of the GraviLogic method: the model
# decomposition f = g∘h, the trajectory γ(t), and a helper that ties the
# two together so counterfactuals.py and core.py agree on what a
# "prediction function" actually means. This last part fixes the
# mismatch we found earlier (build_trajectory wanted a function of z,
# counterfactuals.py wanted a function of x) by making one the
# single source of truth for the other, instead of two independent,
# possibly-inconsistent functions floating around.

from dataclasses import dataclass
# dataclass — a Python decorator that auto-generates __init__ and other
# boilerplate methods for a data-container class, so we don't write them
# by hand.

import numpy as np
# numpy is needed for working with vectors/arrays (latent points, input
# data — all of it is numpy arrays).


class Encoder:
    """h: X -> Z, a wrapper around any model with an explicit latent layer."""
    # This is the code implementation of the function h from the thesis
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
        # entirely different.
        # extract_layer — the function that pulls out the target layer's activations

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

    z_path: np.ndarray   # shape (T, d) — latent points γ(t_i)
    # Array of shape (T, d): T is the number of discretization points,
    # d is the dimensionality of latent space Z.
    # Each row z_path[i] is one point γ(t_i), i.e. the result of
    # passing the i-th intermediate point through the encoder.

    t: np.ndarray        # shape (T,)  — values of parameter t
    # Array of length T with values of parameter t from 0 to 1 —
    # i.e. "how far along the reasoning process" each point z_path[i]
    # is. t[0] = 0 corresponds to the original x, t[-1] = 1 to the
    # counterfactual x'.

    f_path: np.ndarray   # shape (T,)  — predictions g(γ(t_i)) along the path
    # Array of length T: the model's prediction (output of g) at
    # each point of the trajectory. Needed for metrics that look not
    # just at the geometry of z, but at how the prediction itself
    # changes (e.g. Δy — prediction shift, sensitivity).

    @property
    def T(self) -> int:
        # @property lets us access T as an attribute (trajectory.T)
        # rather than as a method (trajectory.T()) — purely for
        # readability.
        return len(self.t)
        # The number of discretization points is just the length of
        # array t.


def build_trajectory(
    encoder: Encoder,
    x: np.ndarray,
    x_prime: np.ndarray,
    predict_from_latent,
    T: int = 50,
) -> Trajectory:
    """γ(0)=h(x), γ(1)=h(x'), linear interpolation in input space (Eq. 9)."""
    # The function that actually BUILDS the trajectory from the thesis
    # for a specific pair (x, x'). Takes:
    # encoder            — an Encoder object (i.e. h)
    # x                  — the original (factual) input
    # x_prime            — the counterfactual input x' (now produced by
    #                       counterfactuals.select_counterfactual, see
    #                       that module for how it's chosen)
    # predict_from_latent — a function z -> prediction, i.e. THIS IS g,
    #                       the decoder half of f = g∘h. Named explicitly
    #                       "from_latent" (rather than the old generic
    #                       "predict_fn") so it can never be confused with
    #                       the full end-to-end f(x) used in
    #                       counterfactuals.py — see compose_predict_from_input()
    #                       below, which builds f(x) FROM this function,
    #                       so you only ever have to define g once.
    # T                  — how many discretization points to use (default 50)

    t = np.linspace(0.0, 1.0, T)
    # np.linspace(0, 1, T) creates T evenly spaced numbers from 0 to 1
    # inclusive. These are the discretization nodes for parameter t.

    x_path = np.array([(1 - ti) * x + ti * x_prime for ti in t])
    # For each ti in t, compute the point on the line between x and x':
    # (1 - ti) * x + ti * x'.
    # At ti=0 we get x, at ti=1 we get x', in between — the
    # intermediate points. This is the equation x(t) = (1-t)x + tx'
    # from the thesis.
    # [... for ti in t] is a list comprehension, a shorthand loop that
    # builds a list directly; np.array(...) turns that list into a
    # single numpy array of shape (T, n), n being the input dimension.

    z_path = np.array([encoder(xi) for xi in x_path])
    # Pass EVERY point of the interpolated input x_path through the
    # encoder (i.e. through h), getting the corresponding latent
    # point z = h(x(t)). The result is an array of all trajectory
    # points γ(t) in latent space, shape (T, d).

    f_path = np.array([predict_from_latent(zi) for zi in z_path])
    # Similarly, for each latent point zi compute the model's
    # prediction predict_from_latent(zi) — i.e. apply g. This gives us
    # how the model's output changes along the whole trajectory.
    # IMPORTANT: this function takes z (latent), NOT x (input) — see the
    # note on predict_from_latent above. Passing a function that expects
    # x here would silently compute garbage (wrong-shaped input to g),
    # so the explicit name is a guardrail, not just documentation.

    return Trajectory(z_path=z_path, t=t, f_path=f_path)
    # Bundle everything into one Trajectory object and return it —
    # this object will be passed into the metric functions (length,
    # curvature, Cognitive Mass, etc.) that come next.


def compose_predict_from_input(encoder: Encoder, predict_from_latent):
    """
    Builds f(x) = g(h(x)) automatically from h (encoder) and g
    (predict_from_latent), so the FULL model function used by
    counterfactuals.py is always derived from — and therefore
    guaranteed consistent with — the exact same g used inside
    build_trajectory(). This removes the earlier inconsistency where
    core.py and counterfactuals.py each expected a differently-shaped
    "predict_fn" defined independently by the user in two places.
    """
    # WHY THIS FUNCTION EXISTS: without it, a user would have to write
    # TWO separate prediction functions by hand — one taking z (for
    # build_trajectory) and one taking x (for select_counterfactual) —
    # and nothing would stop those two from silently disagreeing (e.g.
    # different preprocessing, different rounding, a bug fixed in one
    # but not the other). By deriving the x-based function FROM the
    # z-based one, there is only ONE place where g is defined, and the
    # two modules are mathematically guaranteed to agree, by construction.

    def predict_from_input(x: np.ndarray):
        # This is the function we return: it takes a raw input x (not
        # yet encoded) and returns the model's final prediction f(x).
        return predict_from_latent(encoder(x))
        # encoder(x) computes z = h(x) (calls Encoder.__call__ from above).
        # predict_from_latent(z) then computes g(z).
        # Composed together: predict_from_latent(encoder(x)) = g(h(x)) = f(x),
        # exactly matching the thesis's decomposition f = g∘h (Eq. 7).

    return predict_from_input
    # Return the composed function itself (not its result) — the caller
    # will use this as the `predict_from_input` argument to
    # select_counterfactual() in counterfactuals.py.


