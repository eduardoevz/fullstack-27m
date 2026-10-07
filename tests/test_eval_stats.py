import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.stats import bootstrap_ci, pass_at_k


def test_bootstrap_ci_contains_the_mean_and_is_ordered():
    mean, lo, hi = bootstrap_ci([0, 1, 1, 0, 1, 1, 1, 0, 1, 1], seed=0)
    assert mean == pytest.approx(0.7)
    assert lo <= mean <= hi and 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_is_degenerate_when_all_values_equal():
    assert bootstrap_ci([1, 1, 1, 1], seed=0) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([0, 0, 0], seed=0) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_is_reproducible_and_narrower_with_more_data():
    small = bootstrap_ci([0, 1] * 5, seed=3)
    again = bootstrap_ci([0, 1] * 5, seed=3)
    big = bootstrap_ci([0, 1] * 200, seed=3)
    assert small == again
    assert (big[2] - big[1]) < (small[2] - small[1])


def test_bootstrap_ci_of_empty_input_is_none():
    assert bootstrap_ci([], seed=0) == (None, None, None)


def test_pass_at_k_edge_cases():
    assert pass_at_k(5, 0, 1) == 0.0
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 2, 1) == pytest.approx(0.4)             # con k=1 es c/n


def test_pass_at_k_unbiased_formula():
    # n=5, c=2, k=2:  1 - C(3,2)/C(5,2) = 1 - 3/10
    assert pass_at_k(5, 2, 2) == pytest.approx(0.7)
    assert pass_at_k(5, 4, 3) == 1.0                             # n-c < k  ->  siempre acierta alguno
