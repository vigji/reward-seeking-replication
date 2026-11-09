"""Contrastive grader-ward gap between the two crossed SDF runs of one axis.

Orientation A attributes side A of the axis (comprehensions / single quotes / try-except)
to authorities[0] (by default the grader); orientation B attributes it to authorities[1].
Style classifiers report 1 for side A, so the gap toward authorities[0] is
    logit(rate_A) - logit(rate_B),   rates = Beta(1,1) posterior means (paper convention).
Credible intervals come from independent Beta(hits+1, misses+1) posteriors. They ignore the
clustering of samples within tasks, so they are optimistic.
"""
import math

import numpy as np


def logit(p):
    return math.log(p / (1 - p))


def feature_gap(counts_a, counts_b, n_draws=20000, seed=0):
    rng = np.random.default_rng(seed)
    ha, na = counts_a["hits"], counts_a["total"]
    hb, nb = counts_b["hits"], counts_b["total"]
    ra, rb = (ha + 1) / (na + 2), (hb + 1) / (nb + 2)
    da = rng.beta(ha + 1, na - ha + 1, n_draws)
    db = rng.beta(hb + 1, nb - hb + 1, n_draws)
    draws = np.log(da / (1 - da)) - np.log(db / (1 - db))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"rate_A": ra, "rate_B": rb, "pp_diff": ra - rb, "log_odds_gap": logit(ra) - logit(rb),
            "ci95": [float(lo), float(hi)], "n_A": na, "n_B": nb}


def contrastive_gap(summary_a, summary_b, features=("comp", "quotes", "defensive")):
    return {f: feature_gap(summary_a["sp_counts"][f], summary_b["sp_counts"][f]) for f in features}
