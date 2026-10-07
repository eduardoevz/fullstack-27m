import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.diagnose import bucket_means, classify_hypothesis, doc_spans, window_stats

FILE, EOT = 1, 2


def stream():
    # doc A: 5 tokens (FILE lang x x EOT), doc B: 4 tokens, doc C: 6 tokens
    return np.array([1, 3, 9, 9, 2, 1, 3, 9, 2, 1, 4, 9, 9, 9, 2], dtype=np.uint16)


def test_doc_spans_returns_start_and_end_of_each_framed_document():
    assert doc_spans(stream(), FILE, EOT) == [(0, 4), (5, 8), (9, 14)]


def test_doc_spans_ignores_a_truncated_leading_fragment():
    data = np.concatenate([np.array([9, 9, 2], dtype=np.uint16), stream()])
    assert doc_spans(data, FILE, EOT) == [(3, 7), (8, 11), (12, 17)]


def test_bucket_means_groups_positions():
    losses = np.arange(8, dtype=np.float64)                   # posiciones 0..7
    out = bucket_means(losses, [(0, 2), (2, 5), (5, 8)])
    assert out == {"0-1": 0.5, "2-4": 3.0, "5-7": 6.0}


def test_window_stats_counts_eot_and_starts():
    s = window_stats(stream(), FILE, EOT, window=5)
    n_win = len(stream()) - 5                                   # offsets 0..n-6
    assert s["n_windows"] == n_win
    assert 0.0 < s["share_with_eot"] <= 1.0
    assert s["share_starting_at_doc_start"] == pytest.approx(3 / n_win)  # inicios en 0,5,9


def test_classify_hypothesis_follows_the_preregistered_rule():
    assert classify_hypothesis(median_p_eot=0.1, start_loss_ratio=1.25) == "confirmada"
    assert classify_hypothesis(median_p_eot=0.6, start_loss_ratio=1.5) == "descartada"
    assert classify_hypothesis(median_p_eot=0.1, start_loss_ratio=1.02) == "mixta"
    assert classify_hypothesis(median_p_eot=0.4, start_loss_ratio=1.5) == "mixta"
