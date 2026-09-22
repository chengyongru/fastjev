import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import calibrate  # noqa: E402


def test_bad_temperature_rejected():
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            calibrate.check_temperature(bad)


def test_option_set_mismatch_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text(json.dumps({"id": "x", "family": "f", "group_id": "g", "label": 0,
                                "options": [{"id": "a"}, {"id": "b"}]}) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "c"], "option_logits": [1.0, 2.0]}) + "\n")
    with pytest.raises(ValueError):
        calibrate.load_pairs(gold, pred)


def test_duplicate_gold_row_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    row = {"id": "x", "family": "f", "group_id": "g", "label": 0, "options": [{"id": "a"}, {"id": "b"}]}
    gold.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "b"], "option_logits": [1.0, 2.0]}) + "\n")
    with pytest.raises(ValueError):
        calibrate.load_pairs(gold, pred)


def test_non_finite_logits_rejected(tmp_path):
    gold = tmp_path / "gold.jsonl"
    pred = tmp_path / "pred.jsonl"
    gold.write_text(json.dumps({"id": "x", "family": "f", "group_id": "g", "label": 0,
                                "options": [{"id": "a"}, {"id": "b"}]}) + "\n")
    pred.write_text(json.dumps({"id": "x", "option_ids": ["a", "b"], "option_logits": [1.0, float("inf")]}) + "\n")
    with pytest.raises(ValueError):
        calibrate.load_pairs(gold, pred)


def test_argmax_invariant_under_temperature():
    pairs = calibrate.load_pairs(ROOT / "benchmarks/data/authored144.jsonl",
                                 ROOT / "results/raw/predictions/direct-authored144.jsonl")
    base = calibrate.scored(pairs, 1.0)
    hot = calibrate.scored(pairs, calibrate.fit_temperature(pairs))
    assert [r["correct"] for r in base] == [r["correct"] for r in hot]
