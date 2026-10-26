"""
src/keyword_rules.py - the hand-written keyword-rule classifier.

This lives in its OWN module (not inside src/baselines.py) for one concrete
reason: joblib/pickle saves objects together with the module they were
defined in. A class defined inside a script that runs as __main__ cannot be
loaded back by any other script. Here the class's real home is
"src.keyword_rules", so models/keyword.joblib loads anywhere.

The rules themselves live in configs/keywords.yaml (see that file for the
verdict logic). Matching is case-insensitive substring matching.
"""
from __future__ import annotations


class KeywordRuleBaseline:
    """The hand-written rules from configs/keywords.yaml (see its header)."""

    def __init__(self, cfg: dict):
        self.strong = [w.lower() for w in cfg.get("strong_scam_keywords", [])]
        self.weak = [w.lower() for w in cfg.get("weak_risk_keywords", [])]
        self.veto = [w.lower() for w in cfg.get("safe_veto_phrases", [])]
        self.tactic_kw = {
            tag: [w.lower() for w in words]
            for tag, words in cfg.get("tactics", {}).items()
        }

    def score_one(self, text: str) -> float:
        """A crude 0..1 risk score for AUROC / PR-AUC comparisons.

        Strong rules -> 1.0, safe veto -> 0.0, weak risk -> 0.6, otherwise
        0.0. This is not calibrated; it only lets the rule baseline be ranked
        against probability models. The verdict logic stays exactly as below.
        """
        t = " " + str(text).lower() + " "
        if any(w in t for w in self.strong):
            return 1.0
        if any(w in t for w in self.veto):
            return 0.0
        if any(w in t for w in self.weak):
            return 0.6
        return 0.0

    def predict_one(self, text: str) -> tuple[str, list[str]]:
        """Return (verdict, tactic tags) for one message.

        Logic (identical to the docstring in configs/keywords.yaml):
          1. strong scam word                        -> scam
          2. else real-bank phrase (veto)            -> safe
          3. else weak risk word                     -> suspicious
          4. else                                    -> safe
        """
        t = " " + str(text).lower() + " "
        tags = [tag for tag, words in self.tactic_kw.items()
                if any(w in t for w in words)]
        if any(w in t for w in self.strong):
            return "scam", sorted(tags)
        if any(w in t for w in self.veto):
            return "safe", []
        if any(w in t for w in self.weak):
            return "suspicious", sorted(tags)
        return "safe", []
