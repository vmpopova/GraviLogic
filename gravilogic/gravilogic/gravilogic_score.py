"""gravilogic/score.py — GraviLogic Score (Layer 3 of the method)."""
# WHERE THIS FITS: the method has three layers.
#   Layer 1 (metrics.py) — primitive metrics, each measuring one distinct
#            source of information about a trajectory.
#   Layer 2 (metrics.py) — composite metrics, explicit interactions
#            between primitives from different sources.
#   Layer 3 (THIS FILE) — GraviLogic Score, a single number meant to
#            answer the question "is this model's decision geometrically
#            stable or not, as far as we can currently tell."
#
# STATUS: THIS IS AN EXPERIMENTAL, UNCALIBRATED FORMULA, NOT A FINISHED
# RESULT. It exists so the method HAS a Score-shaped output to iterate on
# and test, not because the weighting below has been validated. The
# individual Layer 1/2 metrics it combines went through real empirical
# testing (see metrics.py's per-function comments — HTRU2, Mann-Whitney U,
# AUC values); the Score itself has NOT yet been tested the same way
# (e.g. checking whether the Score's own AUC/F1 for detecting
# misclassification beats or matches the best individual metric or the
# multi-signal audit rule found so far). Until that test is run, treat
# every number this function produces as illustrative, not as evidence.
#
# WHY A SCORE IS WANTED AT ALL, GIVEN THE AUDIT RULE ALREADY WORKS:
# the OR-rule audit (see metrics.py's module docstring: Manifold Support
# + Bending Energy + Trajectory Deviation Mass, F1=0.654 on HTRU2) answers
# a binary question — "flag this or don't." The Score is meant to answer
# a different, complementary question: a continuous measure of HOW
# geometrically stable a given decision is, for cases where a yes/no flag
# is too coarse (e.g. ranking many predictions by risk, or tracking
# whether a model's stability is drifting over time in production,
# rather than just counting how many predictions get flagged).

from __future__ import annotations


def gravilogic_score(metric_values: dict, calibration_stats: dict, weights: dict = None) -> float:
    """
    metric_values: e.g. {"B": 0.0002, "M": 0.01, "TDM": 4.9, ...} — the
        Layer 1/2 metric values computed for ONE trajectory being scored
        (via metrics.py's functions).
    calibration_stats: {"B": (mean, std), "M": (mean, std), ...} — computed
        from a reference group (e.g. correctly-classified points on a
        held-out set), the same way the audit rule's P25/P75 thresholds
        were calibrated on the correct-group distribution in metrics.py.
    weights: optional per-metric weights summing to 1. Defaults to equal
        weighting — an explicit, labeled placeholder, not a claim that
        equal weighting is the right choice.
    """
    names = list(metric_values.keys())

    if weights is None:
        weights = {k: 1.0 / len(names) for k in names}
        # TEST DEFAULT: equal weight per metric, purely so the formula
        # runs end-to-end. KNOWN ISSUE: local_sensitivity ("S") was found
        # to be INVERTED on HTRU2 — lower values, not higher, associate
        # with misclassification (see metrics.py). Including "S" here
        # with a naive positive weight would subtract from the Score
        # exactly where a real risk should add to it. This function does
        # not fix that for you — do not pass "S" into metric_values until
        # its sign/interpretation is resolved (either exclude it, as the
        # current audit rule does, or explicitly flip its contribution
        # after separately confirming the inversion holds on your data).

    z_scores = {}
    for k in names:
        mean, std = calibration_stats[k]
        z_scores[k] = (metric_values[k] - mean) / std if std > 0 else 0.0
        # Standard z-score: how many standard deviations this
        # trajectory's metric is from the calibration group's mean.
        # Necessary because raw metric scales differ wildly (e.g.
        # Bending Energy ~1e-4 vs Trajectory Deviation Mass ~5) — without
        # normalizing first, whichever metric happens to have the largest
        # raw numbers would dominate the sum regardless of its assigned
        # weight.

    return sum(weights[k] * z_scores[k] for k in names)
    # Weighted sum of normalized deviations. Higher = further from
    # typical calibration-group behavior, i.e. more geometrically
    # anomalous. Again: this produces A NUMBER, not yet a VALIDATED
    # score — that requires the same empirical treatment metrics.py's
    # individual signals received (AUC/F1 against real misclassification
    # or another ground-truth risk label), which hasn't been done for
    # this combined formula yet.


# =============================================================================
# PLANNED — NOT IMPLEMENTED YET. Documenting the intended direction here so
# the roadmap lives next to the code it will eventually extend, rather than
# only in conversation notes.
# =============================================================================
#
# TWO-LEVEL AUDIT REPORT (planned):
#   Technical report — full breakdown of every Layer 1/2 metric value for
#     a given prediction, plus the Score and which calibration thresholds
#     it did/didn't cross. Audience: ML engineers debugging a specific
#     model or decision.
#   Business report — plain-language translation: what kind of risk was
#     detected (e.g. "this decision was made in a region with almost no
#     supporting training data" instead of "Manifold Support = 0.008"),
#     what standard methods analyzing only the model's OUTPUT (SHAP, LIME,
#     accuracy-based monitoring) would have missed and why, and what the
#     practical consequence could be. Audience: people deciding whether to
#     trust or ship a model, who don't need to read a Jacobian norm to
#     act on the finding.
#
# REGULATORY MAPPING (planned, business-report section):
#   The business report could map detected risks to specific articles of
#   the EU AI Act for high-risk AI systems (credit scoring, medical
#   diagnosis, and similar Annex III domains are the method's original
#   target applications):
#     - Art. 13 (Transparency to Deployers) — per-decision explanation of
#       why a prediction was flagged, addressing the requirement that a
#       system's operation be "sufficiently transparent to enable
#       deployers to interpret a system's output."
#     - Art. 9 / Art. 72 (Risk Management System / Post-Market Monitoring)
#       — Manifold Support tracked over incoming data as a continuous,
#       post-deployment drift/OOD signal, not just a one-time
#       certification check.
#     - Art. 15 (Accuracy, Robustness) — Bending Energy / Sensitivity
#       comparisons between models of equal accuracy but different
#       internal stability, surfacing a gap that accuracy-only
#       certification cannot see.
#   IMPORTANT CAVEAT for whenever this section is actually written: the
#   Act does not certify specific XAI techniques or endorse GraviLogic as
#   an official compliance method — any such report section must be
#   framed as "a candidate tool for operationalizing Art. X," never as
#   "this tool ensures compliance with the EU AI Act." The distinction is
#   not stylistic; the stronger phrasing is not legally accurate and would
#   be an easy, fair target for a reviewer or a regulator to challenge.
