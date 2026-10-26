"""
tests/test_phase3.py - the small, fast guarantees of the training code.

These tests do NOT download a pretrained model and do NOT train. They check:
  * label maps round-trip through JSON;
  * collate pads to the BATCH maximum (dynamic padding) and masks padding;
  * the multi-task loss runs on a tiny random batch;
  * the best-checkpoint selector only accepts validation metrics;
  * no message id is ever in two splits.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.config import data_path
from src.data_torch import build_label_maps, load_label_maps, save_label_maps
from src.model import multi_task_loss
from src.train import select_best_checkpoint


def test_label_maps_round_trip(tmp_path):
    maps = build_label_maps()
    path = save_label_maps(tmp_path / "maps.json", maps)
    loaded = load_label_maps(path)
    assert loaded == maps
    assert loaded["verdict_classes"] == ["safe", "scam"]
    assert loaded["num_tags"] == 13
    assert json.loads(path.read_text())["num_verdicts"] == 2


def test_collate_pads_to_batch_max():
    from src.data_torch import make_collate_fn

    collate = make_collate_fn(pad_token_id=0)
    batch = [
        {"input_ids": torch.tensor([5, 6, 7]),
         "attention_mask": torch.ones(3, dtype=torch.long),
         "verdict": torch.tensor(1),
         "tags": torch.zeros(13)},
        {"input_ids": torch.tensor([8, 9, 10, 11, 12]),
         "attention_mask": torch.ones(5, dtype=torch.long),
         "verdict": torch.tensor(0),
         "tags": torch.ones(13)},
    ]
    out = collate(batch)
    assert out["input_ids"].shape == (2, 5)
    assert out["attention_mask"].shape == (2, 5)
    # First row is short: its last two slots are padding (id 0, mask 0).
    assert out["input_ids"][0, 3:].tolist() == [0, 0]
    assert out["attention_mask"][0, 3:].tolist() == [0, 0]
    assert out["verdict"].tolist() == [1, 0]
    assert out["tags"].shape == (2, 13)


def test_loss_runs_on_tiny_random_batch():
    torch.manual_seed(42)
    verdict_logits = torch.randn(4, 2, requires_grad=True)
    tag_logits = torch.randn(4, 13, requires_grad=True)
    verdict_labels = torch.tensor([0, 1, 0, 1])
    tag_labels = (torch.rand(4, 13) > 0.7).float()
    loss, parts = multi_task_loss(
        verdict_logits, tag_logits, verdict_labels, tag_labels,
        verdict_weights=torch.tensor([1.0, 1.5]),
        tag_pos_weight=torch.ones(13), lambda_tags=0.5)
    assert torch.isfinite(loss)
    loss.backward()
    assert verdict_logits.grad is not None
    assert "verdict_loss" in parts


def test_lambda_tags_zero_skips_tag_loss():
    verdict_logits = torch.randn(2, 2)
    tag_logits = torch.randn(2, 13)
    loss, parts = multi_task_loss(
        verdict_logits, tag_logits, torch.tensor([0, 1]),
        torch.zeros(2, 13), lambda_tags=0.0)
    assert parts["tag_loss"] == 0.0
    assert torch.isfinite(loss)


def test_checkpoint_selection_never_reads_test_metrics():
    """Phase 3.6: selection is a MEAN of two validation-only numbers.

    The selector must not even have a test parameter, so selecting on test is
    structurally impossible.
    """
    from src.train import selection_score

    params = list(inspect.signature(selection_score).parameters)
    assert params == ["val_synth_macro_f1", "val_real_balanced_accuracy"]
    assert all(p.startswith("val") for p in params)

    # 0.6 synthetic macro-F1 + 0.8 real balanced accuracy -> 0.7, i.e. a MEAN:
    # the honest real number counts as much as the synthetic one.
    assert selection_score(0.6, 0.8) == pytest.approx(0.7)
    assert selection_score(0.9, None) == pytest.approx(0.9)   # no real val rows
    assert selection_score(None, None) == 0.0

    assert select_best_checkpoint(0.8, 0.7) is True
    assert select_best_checkpoint(0.7, 0.7) is False


def test_selection_prefers_balanced_over_over_flagging():
    """The metric swap must reject the "flag everything" model.

    Real val is 98.3% scam. Under the old combined macro-F1 an always-scam
    model looked great (~0.94); under the new score it does not, because its
    real balanced accuracy is only 0.5.
    """
    from src.train import selection_score

    always_scam = selection_score(val_synth_macro_f1=0.48,
                                  val_real_balanced_accuracy=0.50)
    real_model = selection_score(val_synth_macro_f1=0.93,
                                 val_real_balanced_accuracy=0.85)
    assert real_model > always_scam


def test_no_id_appears_in_two_splits():
    path = data_path("messages_parquet")
    if not path.exists():
        pytest.skip("messages.parquet not built")
    df = pd.read_parquet(path)
    spans = df.groupby("id")["split"].nunique()
    assert (spans == 1).all(), f"ids in two splits: {spans[spans > 1].index[:5].tolist()}"


# ---------------------------------------------------------------------------
# Phase 3 honesty guarantees added after the Phase 2.5 review
# ---------------------------------------------------------------------------
def test_hard_negative_fpr_is_none_when_not_annotated():
    """Real rows carry no hard_negative note; 0/0 must be None, not 0.000."""
    from src.evaluate import verdict_metrics

    df = pd.DataFrame({"verdict": ["safe", "scam"]})
    m = verdict_metrics(df, ["safe", "scam"])
    assert m["hard_negatives"] == 0
    assert m["fpr_on_hard_negatives"] is None
    assert m["fpr_on_safe"] == 0.0


def test_tag_metrics_hide_unvalidated_tags():
    """Per-tag F1 is only a headline number when support >= 20."""
    from src.config import TACTIC_TAGS
    from src.evaluate import tactic_metrics

    df = pd.DataFrame({
        "tactics": [[TACTIC_TAGS[0]], [TACTIC_TAGS[1]], [TACTIC_TAGS[1]]],
    })
    preds = [[TACTIC_TAGS[0]], [TACTIC_TAGS[1]], [TACTIC_TAGS[1]]]
    m = tactic_metrics(df, preds, validated_tags=[TACTIC_TAGS[0]])
    assert m["headline_tags"] == [TACTIC_TAGS[0]]
    assert m["micro_f1"] == 1.0
    # No validated tag -> no headline score at all.
    m2 = tactic_metrics(df, preds, validated_tags=[])
    assert m2["micro_f1"] is None and m2["macro_f1"] is None


def test_models_without_tag_head_are_not_scored():
    from src.evaluate import _empty_tactics, _tag_capable

    assert _tag_capable({"kind": "verdict_lr"}) is False
    assert _tag_capable({"kind": "style_only"}) is False
    assert _tag_capable({"kind": "tactics_lr"}) is True
    assert _empty_tactics()["micro_f1"] is None
    assert _empty_tactics()["available"] is False


def test_e4_gate_thresholds():
    from src.e4_cv import gate_ok

    def frame(n_scam, n_safe):
        strata = (["user_real_scam"] * n_scam) + (["user_real_safe"] * n_safe)
        return pd.DataFrame({"real_stratum": strata})

    assert gate_ok(frame(40, 20))[0] is True
    assert gate_ok(frame(30, 19))[0] is False       # safe hard negatives too few
    assert gate_ok(frame(30, 10))[0] is False       # only 40 in-domain
    ok, n_in, n_safe = gate_ok(frame(5, 5))
    assert (ok, n_in, n_safe) == (False, 10, 5)


def test_train_loop_selects_on_val_only_by_source_inspection():
    """A guard against a future edit that quietly selects on test_*."""
    import src.train as train_module

    source = inspect.getsource(train_module.train)
    assert 'select_best_checkpoint(row["selection_score"]' in source
    assert 'row["val_synth_macro_f1"]' in source
    assert 'row["val_real_balanced_accuracy"]' in source
    assert 'select_best_checkpoint(row["test' not in source
    assert 'select_best_checkpoint(obs' not in source
    assert 'select_best_checkpoint(row["val_macro_f1"]' not in source
    assert "Checkpoint selection used VALIDATION ONLY" in source


def test_class_weights_are_capped_and_finite():
    from src.data_torch import build_label_maps, compute_class_weights

    maps = build_label_maps()
    df = pd.DataFrame({
        "verdict": ["scam"] + ["safe"] * 99,
        "tactics": [[maps["tactic_tags"][0]]] + [[]] * 99,
    })
    verdict_w, tag_pos_w = compute_class_weights(df, maps, max_weight=10.0)
    assert torch.isfinite(verdict_w).all() and torch.isfinite(tag_pos_w).all()
    assert float(verdict_w.max()) <= 10.0
    assert float(tag_pos_w.max()) <= 10.0
    assert tag_pos_w[0] > 1.0          # the rare tag is up-weighted


def test_transformer_predictions_merge_missing_notes_and_skip_untrained_tags(tmp_path, monkeypatch):
    """Regression test for the Phase 3 crash: preds_test_real had no `notes`.

    It also proves a lambda_tags=0 run gets n/a tag metrics, never 0.000.
    """
    import yaml

    import src.config as config_module
    from src.evaluate import add_transformer_runs

    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    run = tmp_path / "outputs" / "e2_test"
    run.mkdir(parents=True)
    (run / "config_used.yaml").write_text(yaml.safe_dump({
        "model_name": "google/muril-base-cased", "lambda_tags": 0.0,
        "checkpoint_name": "best_model.pt"}), encoding="utf-8")
    (run / "training_log.jsonl").write_text(json.dumps({
        "epoch": 1, "val_macro_f1": 0.8, "test_a_macro_f1": 0.7,
        "test_b_macro_f1": 0.6}) + "\n", encoding="utf-8")

    rows = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "verdict": ["scam", "scam", "safe", "safe"],
        "text": ["t1", "t2", "t3", "t4"],
        "tactics": [[], [], [], []],
        "real_stratum": ["user_real_scam", "user_real_scam",
                         "user_real_safe", "user_real_safe"],
        "split": ["test_real"] * 4,
    })
    preds = rows.drop(columns=["tactics"]).assign(
        pred_verdict=["scam", "scam", "safe", "safe"],
        pred_tags=[[]] * 4,
        prob_safe=[0.1, 0.2, 0.9, 0.8], prob_scam=[0.9, 0.8, 0.1, 0.2])
    preds.to_parquet(run / "preds_test_real.parquet", index=False)

    results = {
        "meta": {"runs": {}},
        "test_real": {"strata": {
            "user_real_scam": {"n_rows": 2, "validated_tags": [],
                               "models": {}},
            "user_real_safe": {"n_rows": 2, "validated_tags": [],
                               "models": {}},
        }},
    }
    # `truth` is the main table. Notes are needed for the hard-negative metric.
    truth = rows.assign(notes=["", "", "hard_negative", ""])
    add_transformer_runs(results, truth)

    block = results["test_real"]["strata"]["user_real_safe"]
    assert "e2_test" in block["models"]
    m = block["models"]["e2_test"]
    assert m["tactics"]["available"] is False
    assert m["tactics"]["micro_f1"] is None
    assert m["fpr_on_hard_negatives"] == 0.0     # one annotated hard negative
    assert results["meta"]["runs"]["e2_test"]["tags_trained"] is False
