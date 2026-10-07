import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.harness import Decoding
from src.eval.tune_decoding import (MIN_GAIN, admissible, build_stage_a, build_stage_b, choose, objective)


def test_objective_rewards_valid_syntax_and_penalizes_repetitive_fragments():
    assert objective(strict=0.2, repetitive_share=0.2) == pytest.approx(0.5)
    assert objective(strict=1.0, repetitive_share=0.0) == 1.0
    assert objective(strict=0.0, repetitive_share=1.0) == 0.0


def test_admissible_requires_on_topic_not_to_drop_more_than_5_points_and_no_collapse():
    assert admissible(base_on_topic=0.60, on_topic=0.56, mean_repeat=0.05)
    assert not admissible(base_on_topic=0.60, on_topic=0.50, mean_repeat=0.05)
    assert not admissible(base_on_topic=0.60, on_topic=0.70, mean_repeat=0.0)      # diversidad colapsada


def test_stage_a_changes_one_parameter_at_a_time_and_includes_the_base():
    cfgs = build_stage_a()
    names = [name for name, _ in cfgs]
    assert names[0] == "base" and len(names) == len(set(names)) == 11
    base = Decoding()
    for name, d in cfgs[1:]:
        diff = [k for k, v in d.generate_kwargs().items() if v != base.generate_kwargs()[k]]
        assert len(diff) == 1, (name, diff)


def row(name, strict, rep_share, on_topic, mean_repeat=0.05):
    return {"name": name, "objective": objective(strict, rep_share), "on_topic": on_topic, "mean_repeat": mean_repeat}


def test_choose_picks_the_best_admissible_config_only_if_it_beats_base_by_the_margin():
    base = row("base", 0.2, 0.2, 0.60)
    good = row("rep=1.1", 0.4, 0.05, 0.60)
    assert choose([base, good])["name"] == "rep=1.1"
    barely = row("rep=1.05", 0.2 + 2 * (MIN_GAIN / 2) - 0.01, 0.2, 0.60)             # mejora < margen
    assert choose([base, barely])["name"] == "base"


def test_choose_ignores_configs_that_break_the_constraints():
    base = row("base", 0.2, 0.2, 0.60)
    bad_topic = row("ngram=3", 0.9, 0.0, 0.30)
    assert choose([base, bad_topic])["name"] == "base"


def test_stage_b_combines_the_winners_of_each_family():
    stage_a = [row("base", 0.2, 0.2, 0.6),
               {**row("rep=1.1", 0.4, 0.1, 0.6), "family": "rep", "decoding": Decoding(repetition_penalty=1.1).generate_kwargs()},
               {**row("ngram=4", 0.35, 0.1, 0.6), "family": "ngram", "decoding": Decoding(no_repeat_ngram_size=4).generate_kwargs()},
               {**row("minp=0.05", 0.15, 0.3, 0.6), "family": "minp", "decoding": Decoding(min_p=0.05).generate_kwargs()}]
    stage_a[0]["family"] = "base"
    combos = build_stage_b(stage_a)
    kw = {name: d.generate_kwargs() for name, d in combos}
    assert any(v["repetition_penalty"] == 1.1 and v["no_repeat_ngram_size"] == 4 for v in kw.values())
    assert not any(v["min_p"] != 0.0 for v in kw.values())                         # minp no mejoro: no entra
    assert 1 <= len(combos) <= 4
