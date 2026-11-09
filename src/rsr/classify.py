"""Rollout classifiers (style features, recall answers, Beta-smoothed rates): alphaXiv's src/classify.py,
used unchanged. Fetched by scripts/fetch_alphaxiv.sh into rsr/alphaxiv/ (no upstream license, so not redistributed)."""
from rsr.alphaxiv.classify import classify_recall, classify_style, extract_code, rate_with_beta  # noqa: F401
