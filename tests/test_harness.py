import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.harness import Decoding, summarize_prompts


def sample(pid, strict, trimmed, topic):
    return {"id": pid, "strict_ok": strict, "trimmed_ok": trimmed, "on_topic": topic}


def test_decoding_defaults_equal_the_v1_settings_and_expose_generate_kwargs():
    d = Decoding()
    assert (d.temperature, d.top_k, d.top_p) == (0.8, 40, 0.95)
    kw = d.generate_kwargs()
    assert kw["repetition_penalty"] == 1.0 and kw["no_repeat_ngram_size"] == 0 and kw["min_p"] == 0.0
    assert Decoding(temperature=0.6, min_p=0.05).generate_kwargs()["min_p"] == 0.05


def test_summarize_prompts_averages_samples_within_each_prompt_first():
    rows = [sample("a", True, True, True), sample("a", False, True, True),      # a: 0.5 / 1 / 1
            sample("b", False, False, False), sample("b", False, False, False)]  # b: 0 / 0 / 0
    s = summarize_prompts(rows, seed=0)
    assert s["n_prompts"] == 2 and s["n_samples"] == 4
    assert s["strict"]["mean"] == pytest.approx(0.25)
    assert s["trimmed"]["mean"] == pytest.approx(0.5)
    assert s["on_topic"]["mean"] == pytest.approx(0.5)
    for k in ("strict", "trimmed", "on_topic"):
        assert s[k]["lo"] <= s[k]["mean"] <= s[k]["hi"]


def test_summarize_prompts_is_reproducible():
    rows = [sample(str(i % 7), i % 2 == 0, True, i % 3 == 0) for i in range(35)]
    assert summarize_prompts(rows, seed=1) == summarize_prompts(rows, seed=1)
