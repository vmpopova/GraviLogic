"""gravilogic/counterfactuals.py — axiom-constrained counterfactual selection."""
# This module replaces the plain "nearest opposite-class instance" heuristic
# (still kept in utils.py for comparison/ablation) with a mathematically
# grounded procedure: instead of just grabbing the closest real point with a
# different label, we SOLVE AN OPTIMIZATION PROBLEM whose constraints are
# built directly from the axioms discussed (contrastivity, achievability,
# interpretability, directionality, relevance). This is the Wachter-style
# counterfactual (Wachter, Mittelstadt, Russell 2017), extended with a
# density constraint and a relevance-based feature mask.
#
# NAMING NOTE (reconciled with core.py): every function here that touches
# the model works purely in INPUT space and is named predict_from_input,
# to match exactly the function produced by core.py's
# compose_predict_from_input(). This guarantees the x' this module finds
# is evaluated by the SAME f = g∘h that build_trajectory() will later use
# to build γ(t) — no risk of the two modules silently disagreeing about
# what "the model's prediction" means.
#
# OPTIMIZER NOTE (fixed after an empirical check): the optimizer below uses
# Powell's method, NOT Nelder-Mead. A real test run (sklearn MLP, 30-feature
# tabular dataset) showed Nelder-Mead reporting success=False on every one
# of 25 counterfactual searches, even at 2000 iterations — a known failure
# mode of Nelder-Mead above roughly 10 dimensions. Powell converged
# (success=True) on the identical problem, reaching essentially the same
# objective value. Powell is still derivative-free (matches our need to
# treat predict_from_input and the KDE density as black boxes), just more
# reliable at this dimensionality.

from __future__ import annotations
# Lets us write type hints like "np.ndarray | None" even on Python versions
# where that union syntax wasn't natively supported yet — purely a
# compatibility import, has no effect on the logic.

import numpy as np
# numpy for vector/array math — x, x', gradients are all numpy arrays.

from scipy.optimize import minimize
# scipy's general-purpose numerical optimizer. We use it to actually SOLVE
# the counterfactual search instead of hand-rolling gradient descent —
# minimize() handles step sizes, convergence checks, etc. for us.

from scipy.stats import gaussian_kde
# Kernel Density Estimation: a standard, well-established statistical way
# to estimate "how plausible/dense is this point relative to real data"
# without assuming any particular distribution shape. This is what lets us
# turn the vague axiom "x' must be realistic" into an actual number.


class DensityEstimator:
    """
    Wraps a KDE fit once on the training distribution, reused across
    every counterfactual search (fitting KDE per-call would be wasteful
    and would make density_threshold uninterpretable across calls).
    """
    # WHY THIS CLASS EXISTS: the "achievability axiom" says x' must belong
    # to X_valid — the region of realistic data. KDE gives us a concrete,
    # reusable, numeric definition of "realistic": a density function
    # rho(x) estimated from real training data. High rho(x) = looks like
    # real data; low rho(x) = looks synthetic/implausible.

    def __init__(self, train_data: np.ndarray, bandwidth: str | float = "scott"):
        # __init__ = constructor, runs once when you create a
        # DensityEstimator(train_data) object.
        # train_data: shape (N, n_features), the real data distribution
        # we require counterfactuals to stay plausible with respect to.

        self._kde = gaussian_kde(train_data.T, bw_method=bandwidth)
        # gaussian_kde expects data as (n_features, N) — the opposite
        # orientation from our usual (N, n_features) convention — hence
        # the .T (transpose). bw_method="scott" is a standard automatic
        # rule for choosing the KDE's smoothing bandwidth, so we don't
        # have to hand-tune it. Storing the fitted KDE as self._kde means
        # we compute it ONCE, not every time density() is called.

    def density(self, x: np.ndarray) -> float:
        """rho(x): estimated density at a single point x, shape (n,)."""
        return float(self._kde(x.reshape(-1, 1))[0])
        # gaussian_kde expects a batch of points shaped (n_features, n_points).
        # x.reshape(-1, 1) turns our single point (shape (n,)) into shape
        # (n, 1) — "one point, n features" — as the KDE API expects.
        # The KDE call returns an array of densities (one per input point);
        # since we passed exactly one point, we take element [0].
        # float(...) converts it from a numpy scalar to a plain Python float.
        #
        # PRACTICAL NOTE (curse of dimensionality): KDE degrades in high
        # dimensions. In the empirical test run, density was computed on
        # only the first 10 features of a 30-feature dataset to keep the
        # estimate tractable — if your input dimensionality is large,
        # consider fitting this on a lower-dimensional projection (e.g. a
        # PCA of the input, or the latent z instead of raw x) rather than
        # the full raw feature space.

    def calibrate_threshold(self, train_data: np.ndarray, percentile: float = 10.0) -> float:
        """
        Suggests rho_min as a low percentile of density over the training
        set itself, so 'achievable' means 'at least as plausible as the
        least-dense 90% of real data' rather than an arbitrary constant.
        """
        # WHY THIS EXISTS: "density_threshold" can't just be a magic number
        # like 0.01 — that's meaningless without knowing the scale of rho
        # for THIS dataset. Instead, we measure the density of every real
        # training point, and set the threshold at a low percentile of
        # THOSE values. That way "achievable" is defined relative to the
        # data itself, not an arbitrary guess.

        densities = np.array([self.density(x) for x in train_data])
        # Compute rho(x) for every single training point — this is a list
        # comprehension (a compact loop) turned into a numpy array.
        # densities[i] = how plausible training point i is, according to
        # the very same KDE that will judge our counterfactuals.

        return float(np.percentile(densities, percentile))
        # np.percentile(densities, 10) returns the value below which 10%
        # of training points fall. Using this as rho_min means: "the
        # counterfactual must be at least as plausible as the least
        # plausible 10% of real data" — a principled, data-driven
        # threshold instead of a guessed constant.


def relevant_features(predict_grad_from_input, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """
    Automatic relevance filter (relevance axiom): returns indices i where
    |∂f/∂x_i| > eps at x, i.e. features the model is actually sensitive to
    at this point. Used to build allowed_features when the caller hasn't
    specified a directionality mask explicitly.

    predict_grad_from_input: callable x -> gradient vector df/dx at x
    (shape (n,)) of the FULL model f = g∘h — e.g. computed via autograd
    on the composition produced by core.py's compose_predict_from_input(),
    or via a closed-form expression if the model is simple enough (as in
    the empirical test run, which used a closed-form ReLU-MLP gradient).
    """
    # WHY THIS EXISTS: a counterfactual should only change features the
    # model actually reacts to (∂f/∂x_i ≠ 0) — moving a feature the model
    # ignores would waste the optimizer's search AND make the resulting
    # trajectory meaningless (moving through a direction that has zero
    # effect on f says nothing about the model's internal logic).
    #
    # PRACTICAL NOTE from the empirical run: on a dense 30-feature MLP,
    # EVERY feature had a nonzero gradient (the mask matched all 30
    # features) — dense architectures often don't give you dimensionality
    # reduction "for free" this way. Don't rely on this filter alone to
    # keep the optimization problem low-dimensional; that's a separate
    # concern from correctness (see the optimizer note at the top of the
    # file — Powell was chosen specifically to remain reliable even when
    # this filter doesn't reduce dimensionality).

    grad = predict_grad_from_input(x)
    # Call the user-supplied gradient function to get ∂f/∂x at the point x.

    return np.where(np.abs(grad) > eps)[0]
    # np.abs(grad) — take the absolute value of every gradient component
    # (we care about MAGNITUDE of sensitivity, not direction, here).
    # np.where(condition) returns the INDICES where the condition is True;
    # [0] extracts just the array of indices (np.where returns a tuple).
    # eps is a small threshold (not exactly 0) because floating-point
    # gradients are almost never EXACTLY zero even when a feature is
    # effectively irrelevant — this avoids false positives from numerical
    # noise. eps=1e-4 (looser than a bare numerical-precision threshold)
    # was used in the empirical run to avoid flagging negligibly small
    # but nonzero gradients as "relevant."


def select_counterfactual(
    x: np.ndarray,
    predict_from_input,
    y_target: float,
    density_estimator: DensityEstimator,
    density_threshold: float,
    predict_grad_from_input=None,
    allowed_features: np.ndarray | None = None,
    l1_weight: float = 1.0,
    contrast_weight: float = 10.0,
    density_penalty_weight: float = 5.0,
    max_iter: int = 300,
    n_starts: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, int, int]:
    """
    Returns (x_prime, n_success, n_starts) where x_prime satisfies (to the
    extent the optimizer converges):
        - contrastivity:      f(x') ~= y_target
        - achievability:      density(x') >= density_threshold
        - interpretability:   ||x' - x||_1 is minimized (sparse Δx)
        - directionality:     only allowed_features may change
        - relevance:          allowed_features defaults to features with
                               nonzero local gradient, if not given explicitly

    n_success / n_starts report how many of the n_starts random restarts
    reported a converged (success=True) result — use this to monitor
    optimizer reliability on your actual data, the same way the earlier
    empirical check did (which is what caught the Nelder-Mead failure
    this function was rewritten to fix).

    predict_from_input: x -> scalar model output f(x) = g(h(x)). Pass in
        core.py's compose_predict_from_input(encoder, g) here so this
        module is guaranteed to score x' with the SAME f that
        build_trajectory() will later use.
    predict_grad_from_input: x -> gradient df/dx, shape (n,). Required only
        if allowed_features is None (used to auto-derive the relevance mask).
    """
    # This is the main function of the module: it actually SEARCHES FOR x'
    # by solving an optimization problem, rather than just picking an
    # existing data point. Every argument below maps to one piece of the
    # axiom system discussed earlier.

    n = x.shape[0]
    # n = number of input features (dimensionality of x). We'll need this
    # to build masks of the right size below.

    if allowed_features is None:
        # If the caller didn't explicitly say WHICH features are allowed
        # to change (directionality axiom), we derive that mask
        # automatically using the relevance axiom instead — i.e. "if you
        # don't tell me what aspect to investigate, default to whatever
        # the model is actually sensitive to here."

        if predict_grad_from_input is None:
            raise ValueError(
                "allowed_features not given and no predict_grad_from_input provided "
                "to derive the relevance mask automatically — supply one or the other."
            )
            # Fail loudly and early rather than silently guessing — if
            # neither a mask nor a gradient function is given, we have no
            # principled way to decide which features to move, and
            # continuing would produce an arbitrary, unjustified result.

        allowed_features = relevant_features(predict_grad_from_input, x)
        # Use the function defined above to get the indices of features
        # with non-negligible local gradient at x.

        if allowed_features.size == 0:
            raise ValueError(
                "No features with nonzero local gradient at x — model is locally "
                "flat here, a meaningful counterfactual cannot be defined."
            )
            # If EVERY feature has zero gradient at this point, the model
            # simply doesn't respond to local changes here at all — there
            # is no honest way to build an informative counterfactual, so
            # we refuse rather than return a meaningless result.

    def unpack(free_vars: np.ndarray) -> np.ndarray:
        # Reconstruct full x' from the free (allowed) variables only;
        # fixed features are pinned to their original x_i (directionality axiom).

        x_prime = x.copy()
        # Start from a COPY of the original x (never mutate x itself) —
        # every feature begins pinned to its original value.

        x_prime[allowed_features] = free_vars
        # Overwrite ONLY the allowed feature positions with whatever
        # values the optimizer is currently trying. Every other position
        # stays exactly equal to x — this physically ENFORCES the
        # directionality axiom: the optimizer literally has no way to
        # change a fixed feature, because unpack() ignores any such
        # attempt by construction.

        return x_prime
        # Return the full-length candidate counterfactual point.

    def objective(free_vars: np.ndarray) -> float:
        # This is the function the optimizer will try to MINIMIZE.

        x_prime = unpack(free_vars)
        # Turn the optimizer's current guess (only for allowed features)
        # into a full candidate point x'.

        l1_term = np.sum(np.abs(x_prime - x))
        # INTERPRETABILITY axiom, operationalized: sum of absolute
        # differences between x' and x (the L1 norm of Δx). L1 (rather
        # than L2/squared distance) is the deliberate choice — L1
        # optimization tends to produce SPARSE solutions, i.e. changes
        # concentrated in a few features rather than spread thinly across
        # all of them, which is what makes Δx interpretable.

        contrast_term = (predict_from_input(x_prime) - y_target) ** 2
        # CONTRASTIVITY axiom, operationalized: squared difference between
        # the model's prediction at the candidate x' and the desired
        # target prediction y_target. Zero exactly when f(x') = y_target.

        rho = density_estimator.density(x_prime)
        # ACHIEVABILITY axiom, step 1: how plausible is this candidate
        # according to the real data distribution?

        density_term = max(0.0, density_threshold - rho)
        # ACHIEVABILITY axiom, step 2: a "hinge" penalty. Zero once rho
        # clears the threshold; grows the further below threshold rho falls.

        return (
            l1_weight * l1_term
            + contrast_weight * contrast_term
            + density_penalty_weight * density_term
        )
        # Combine all three concerns into ONE number the optimizer can
        # minimize, each scaled by its own weight — hyperparameters to
        # tune per dataset, not universal constants.

    rng = np.random.default_rng(seed)
    # A seeded random generator — makes the multi-start restarts below
    # reproducible (same seed => same sequence of starting perturbations),
    # which matters for debugging and for reporting results honestly.

    x0_free = x[allowed_features].copy()
    # The base starting point for optimization: the original values of
    # just the allowed features.

    best_result = None
    n_success = 0
    # We'll run the optimizer n_starts times from slightly different
    # starting points and keep the best one — this is the MULTI-START
    # strategy added after the empirical check showed a single run can
    # get stuck without reporting failure clearly; running several and
    # tracking how many actually report success gives us both a better
    # answer AND a diagnostic signal (n_success / n_starts) about how
    # reliable the optimization is on this particular model/point.

    for i in range(n_starts):
        start = x0_free if i == 0 else x0_free + rng.normal(scale=0.05, size=x0_free.shape)
        # The first restart starts exactly at x (the "natural" starting
        # point); subsequent restarts add small random noise (std=0.05,
        # a small perturbation relative to standardized features) to
        # explore slightly different regions of the search space and
        # avoid all restarts converging to the exact same local optimum.

        result = minimize(objective, start, method="Powell", options={"maxiter": max_iter})
        # Powell's method: derivative-free (matches our need to treat
        # predict_from_input and KDE density as black boxes, same
        # requirement Nelder-Mead was chosen for originally), but reliable
        # at higher dimensionality — confirmed empirically after
        # Nelder-Mead failed to converge (success=False) on a 30-feature
        # problem where Powell succeeded on the identical objective.

        if result.success:
            n_success += 1
        # Track how many of the n_starts restarts scipy itself considers
        # converged — this is the diagnostic returned to the caller.

        if best_result is None or result.fun < best_result.fun:
            best_result = result
        # Keep whichever restart achieved the LOWEST objective value overall,
        # regardless of its individual success flag — a good result that
        # narrowly misses scipy's strict tolerance is still usually more
        # useful than discarding it entirely.

    x_prime = unpack(best_result.x)
    # best_result.x holds the best-found values for the free variables
    # across all restarts; unpack() turns that back into a full-length x'
    # with the fixed features restored to their original values.

    return x_prime, n_success, n_starts
    # Return the final counterfactual point AND the convergence diagnostic
    # (n_success out of n_starts) — the caller (or an experiment script,
    # as in the empirical check) should log this across many calls to
    # monitor whether the optimizer is behaving reliably on real data,
    # not just assume it always works.
