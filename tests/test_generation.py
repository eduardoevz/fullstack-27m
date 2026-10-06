import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig
from src.model.gpt import GPT
from src.sample import filter_logits, generate, sample_next

TINY = ModelConfig(vocab_size=128, n_layer=2, n_head=4, d_model=64, d_ff=128, block_size=32,
                   rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)


def tiny(seed=0):
    torch.manual_seed(seed)
    m = GPT(TINY).eval()
    for p in m.parameters():                       # pesos con mas senal que el init de 0.02
        p.data.mul_(8)
    return m


# ---------------------------------------------------------------- cache KV
def test_cached_prefill_matches_full_forward():
    model = tiny()
    x = torch.randint(0, 128, (2, 20))
    with torch.no_grad():
        full, _ = model(x)
        cached, cache = model.forward_cached(x, None)
    assert torch.allclose(full, cached, atol=1e-5)
    assert len(cache) == TINY.n_layer and cache[0][0].shape[2] == 20


def test_cached_stepwise_matches_full_forward():
    model = tiny()
    x = torch.randint(0, 128, (2, 24))
    with torch.no_grad():
        full, _ = model(x)
        _, cache = model.forward_cached(x[:, :10], None)
        steps = []
        for t in range(10, 24):
            logits, cache = model.forward_cached(x[:, t:t + 1], cache)
            steps.append(logits[:, 0])
    assert torch.allclose(full[:, 10:], torch.stack(steps, dim=1), atol=1e-3)
    assert cache[0][0].shape[2] == 24


def test_cached_chunk_after_prefix_matches_full_forward():
    model = tiny()
    x = torch.randint(0, 128, (1, 18))
    with torch.no_grad():
        full, _ = model(x)
        _, cache = model.forward_cached(x[:, :7], None)
        tail, _ = model.forward_cached(x[:, 7:], cache)
    assert torch.allclose(full[:, 7:], tail, atol=1e-3)


# ---------------------------------------------------------------- muestreo
def test_filter_top_k_keeps_exactly_k():
    logits = torch.arange(10, dtype=torch.float32)
    out = filter_logits(logits, top_k=3, top_p=1.0)
    assert torch.isfinite(out).sum() == 3 and torch.isfinite(out[-3:]).all()


def test_filter_top_p_keeps_smallest_set_reaching_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    out = filter_logits(probs.log(), top_k=0, top_p=0.7)      # 0.5 < 0.7 <= 0.8
    assert torch.isfinite(out).tolist() == [True, True, False, False]


def test_filter_top_p_always_keeps_the_best_token():
    out = filter_logits(torch.tensor([5.0, 0.0, 0.0]), top_k=0, top_p=0.01)
    assert torch.isfinite(out).tolist() == [True, False, False]


def test_sample_next_zero_temperature_is_argmax():
    logits = torch.tensor([0.1, 3.0, 0.2, 2.9])
    assert sample_next(logits, temperature=0.0, top_k=0, top_p=1.0) == 1


def test_sample_next_is_seeded_and_respects_top_k():
    logits = torch.randn(50)
    allowed = set(logits.topk(5).indices.tolist())
    draws = [sample_next(logits, 1.0, 5, 1.0, torch.Generator().manual_seed(s)) for s in range(200)]
    assert set(draws) <= allowed and len(set(draws)) > 1
    a = sample_next(logits, 1.0, 0, 1.0, torch.Generator().manual_seed(7))
    b = sample_next(logits, 1.0, 0, 1.0, torch.Generator().manual_seed(7))
    assert a == b


# ---------------------------------------------------------------- generate
def naive_greedy(model, prompt, n):
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(torch.tensor([ids]))
            ids.append(int(logits[0, -1].argmax()))
    return ids[len(prompt):]


def test_generate_greedy_with_cache_equals_without_cache():
    model = tiny()
    prompt = [3, 9, 27, 81]
    got = generate(model, prompt, max_new_tokens=15, temperature=0.0)
    assert got == naive_greedy(model, prompt, 15)


def test_generate_stops_at_eos_and_excludes_it():
    model = tiny()
    prompt = [3, 9]
    free = generate(model, prompt, max_new_tokens=10, temperature=0.0)
    eos = free[4]
    got = generate(model, prompt, max_new_tokens=10, temperature=0.0, eos_id=eos)
    assert got == free[:free.index(eos)]


def test_generate_respects_max_new_tokens():
    assert len(generate(tiny(), [1, 2, 3], max_new_tokens=7, temperature=1.0, top_k=10,
                        generator=torch.Generator().manual_seed(0))) == 7


def test_generate_is_reproducible_with_a_seeded_generator():
    model = tiny()
    a = generate(model, [1, 2], 12, 1.0, 20, 0.9, generator=torch.Generator().manual_seed(3))
    b = generate(model, [1, 2], 12, 1.0, 20, 0.9, generator=torch.Generator().manual_seed(3))
    assert a == b


def test_generate_goes_past_block_size_with_a_sliding_window():
    model = tiny()
    out = generate(model, [1, 2, 3], max_new_tokens=TINY.block_size * 2, temperature=1.0, top_k=10,
                   generator=torch.Generator().manual_seed(0))
    assert len(out) == TINY.block_size * 2


def test_generate_truncates_a_prompt_longer_than_the_window():
    model = tiny()
    out = generate(model, list(range(100)) * 2, max_new_tokens=5, temperature=0.0)
    assert len(out) == 5


def test_generate_rejects_empty_prompt():
    with pytest.raises(ValueError):
        generate(tiny(), [], max_new_tokens=3)


def test_generate_calls_on_token_for_every_new_token_in_order():
    model = tiny()
    seen = []
    out = generate(model, [1, 2], max_new_tokens=9, temperature=0.0, on_token=seen.append)
    assert seen == out and len(seen) == 9
