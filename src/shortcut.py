"""
src/shortcut.py - metadata-only baseline models (style / source shortcuts).

WHY THIS FILE EXISTS: joblib pickles a class by its MODULE name. If the class
lived inside src/baselines.py and that file was run with `python -m`, it would
be saved as `__main__.SingleFeatureShortcut` and could not be loaded by
src/evaluate.py. Keeping it here gives it a stable name:
`src.shortcut.SingleFeatureShortcut`.

A shortcut model sees ONE metadata column and NO message text. If it scores
well, the DATA leaks the answer (e.g. "all devanagari messages are scams").
"""
from __future__ import annotations

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import SEED


class SingleFeatureShortcut:
    """Predict scam/safe from a single string column (e.g. script_style)."""

    def __init__(self, feature: str):
        self.feature = feature

    def fit(self, values: list[str], y: list[str]) -> "SingleFeatureShortcut":
        self.majority_ = max(set(y), key=y.count)
        # A column with one value (or one label) cannot train a classifier.
        if len(set(values)) < 2 or len(set(y)) < 2:
            self.pipeline_ = None
            return self
        self.pipeline_ = Pipeline([
            ("onehot", DictVectorizer(sparse=False)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=SEED)),
        ])
        self.pipeline_.fit([{self.feature: v} for v in values], y)
        return self

    def predict(self, values: list[str]) -> list[str]:
        if self.pipeline_ is None:
            return [self.majority_] * len(values)
        return list(self.pipeline_.predict([{self.feature: v} for v in values]))

    def predict_proba(self, values: list[str]) -> list[float]:
        """P(scam) for each value, for AUROC / PR-AUC.

        A single-valued column has no classifier; it returns a constant score,
        and evaluate.py will report its ranking metrics as n/a rather than
        inventing an AUROC.
        """
        if self.pipeline_ is None:
            return [1.0 if self.majority_ == "scam" else 0.0] * len(values)
        proba = self.pipeline_.predict_proba([{self.feature: v} for v in values])
        classes = list(self.pipeline_.named_steps["clf"].classes_)
        if "scam" not in classes:
            return [0.0] * len(values)
        idx = classes.index("scam")
        return [float(row[idx]) for row in proba]
