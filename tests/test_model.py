import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, load_config
from src.model.gpt import GPT, CausalSelfAttention, RMSNorm
from src.model.rope import RotaryEmbedding, apply_rotary, build_rope_cache

TINY = ModelConfig(vocab_size=128, n_layer=2, n_head=4, d_model=64, d_ff=128, block_size=32,
                   rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(TINY).eval()


# ---------------------------------------------------------------- parametros y estructura
def test_param_count_matches_analytic_formula():
    cfg = load_config().model
    model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters())          # parameters() no duplica el peso atado
    assert n == cfg.param_count() == 20_453_760


def test_no_biases_and_tied_embeddings():
    model = tiny()
    assert not [n for n, _ in model.named_parameters() if n.endswith("bias")]
    assert model.lm_head.weight is model.tok_emb.weight


def test_shapes_and_optional_loss():
    model = tiny()
    x = torch.randint(0, 128, (3, 16))
    logits, loss = model(x)
    assert logits.shape == (3, 16, 128) and loss is None
    logits, loss = model(x, targets=x)
    assert loss.ndim == 0 and torch.isfinite(loss)


# ---------------------------------------------------------------- comportamiento
def test_causality():
    model = tiny()
    x = torch.randint(0, 128, (2, 24))
    y = x.clone()
    t = 10
    y[:, t] = (y[:, t] + 1) % 128
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(y)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)
    assert not torch.allclose(a[:, t:], b[:, t:], atol=1e-6)


def test_determinism_with_same_seed():
    x = torch.randint(0, 128, (2, 16))
    with torch.no_grad():
        a, _ = tiny(seed=5)(x)
        b, _ = tiny(seed=5)(x)
        c, _ = tiny(seed=6)(x)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_initial_loss_is_ln_vocab_on_real_config():
    cfg = load_config().model
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    with torch.no_grad():
        _, loss = model(x, targets=torch.randint(0, cfg.vocab_size, (2, 128)))
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.25       # ln(16384) = 9.70


def test_gradients_flow_to_every_parameter():
    model = tiny().train()
    x = torch.randint(0, 128, (4, 16))
    _, loss = model(x, targets=x)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = GPT(TINY).train()
    x = torch.randint(0, 128, (32, 32))
    y = torch.randint(0, 128, (32, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    loss = None
    for _ in range(800):
        _, loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.05:
            break
    assert loss.item() < 0.1


def test_longer_than_block_size_still_runs():
    model = tiny()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, 128, (1, TINY.block_size + 8)))
    assert logits.shape == (1, TINY.block_size + 8, 128)


# ---------------------------------------------------------------- RoPE
def test_rope_preserves_norm():
    cos, sin = build_rope_cache(16, 32, 10000.0)
    x = torch.randn(2, 4, 32, 16)
    y = apply_rotary(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-5)


def test_rope_score_depends_only_on_relative_distance():
    cos, sin = build_rope_cache(16, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)

    def score(m, n):
        qm = apply_rotary(q, cos[m:m + 1], sin[m:m + 1])
        kn = apply_rotary(k, cos[n:n + 1], sin[n:n + 1])
        return (qm * kn).sum().item()

    assert score(5, 2) == pytest.approx(score(25, 22), abs=1e-4)
    assert score(5, 2) != pytest.approx(score(5, 4), abs=1e-3)


def test_rotary_offset_equals_slicing_and_extends():
    rope = RotaryEmbedding(16, max_len=8, theta=10000.0)
    cos_full, sin_full = rope(7)
    cos_off, sin_off = rope(4, offset=3)
    assert torch.equal(cos_off, cos_full[3:7]) and torch.equal(sin_off, sin_full[3:7])
    cos_long, _ = rope(20)                                       # mas alla de max_len: se extiende
    assert cos_long.shape[0] == 20 and torch.equal(cos_long[:7], cos_full)


# ---------------------------------------------------------------- atencion y norma
def test_attention_matches_manual_masked_softmax():
    torch.manual_seed(0)
    attn = CausalSelfAttention(TINY).eval()
    rope = RotaryEmbedding(TINY.head_dim, TINY.block_size, TINY.rope_theta)
    x = torch.randn(2, 12, TINY.d_model)
    cos, sin = rope(12)
    with torch.no_grad():
        got = attn(x, cos, sin)
        B, T, C = x.shape
        q, k, v = attn.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, TINY.n_head, TINY.head_dim).transpose(1, 2) for t in (q, k, v))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(TINY.head_dim)
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        want = attn.proj((F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, T, C))
    assert torch.allclose(got, want, atol=1e-5)


def test_rmsnorm_matches_formula_and_unit_rms():
    norm = RMSNorm(64, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(64) + 0.5)
    x = torch.randn(3, 5, 64) * 7
    want = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    assert torch.allclose(norm(x), want, atol=1e-5)
    unit = RMSNorm(64, eps=1e-5)
    assert unit(x).pow(2).mean(-1).sqrt().mean().item() == pytest.approx(1.0, abs=1e-3)
