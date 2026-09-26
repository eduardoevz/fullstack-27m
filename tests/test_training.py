import copy
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ModelConfig, TrainingConfig, load_config
from src.model.gpt import GPT
from src.training.checkpoint import (load_checkpoint, rotate_checkpoints, save_checkpoint)
from src.training.data import BatchSampler
from src.training.loop import accumulate_gradients, run_step
from src.training.optim import build_optimizer
from src.training.schedule import get_lr

TINY = ModelConfig(vocab_size=128, n_layer=2, n_head=4, d_model=64, d_ff=128, block_size=32,
                   rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, init_std=0.02)


def tcfg(**over) -> TrainingConfig:
    base = replace(load_config().training, micro_batch_size=2, grad_accum_steps=2, max_steps=100,
                   warmup_steps=10, learning_rate=1e-3, min_learning_rate=1e-4)
    return replace(base, **over)


@pytest.fixture
def bin_file(tmp_path):
    p = tmp_path / "toy.bin"
    np.random.default_rng(0).integers(0, 128, size=5000, dtype=np.uint16).tofile(p)
    return p


# ---------------------------------------------------------------- LR
def test_lr_warmup_peak_cosine_and_floor():
    c = tcfg()
    assert get_lr(0, c) == pytest.approx(c.learning_rate / c.warmup_steps)
    assert get_lr(c.warmup_steps - 1, c) == pytest.approx(c.learning_rate)
    assert get_lr(c.warmup_steps, c) == pytest.approx(c.learning_rate)
    mid = c.warmup_steps + (c.max_steps - c.warmup_steps) // 2
    assert get_lr(mid, c) == pytest.approx((c.learning_rate + c.min_learning_rate) / 2, rel=1e-3)
    assert get_lr(c.max_steps, c) == pytest.approx(c.min_learning_rate)
    assert get_lr(c.max_steps + 50, c) == pytest.approx(c.min_learning_rate)


def test_lr_monotone_after_warmup():
    c = tcfg()
    lrs = [get_lr(s, c) for s in range(c.warmup_steps, c.max_steps + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


# ---------------------------------------------------------------- optimizador
def test_weight_decay_groups():
    model = GPT(TINY)
    opt = build_optimizer(model, tcfg())
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert all(p.ndim >= 2 for p in decay["params"])
    assert id(model.tok_emb.weight) in {id(p) for p in no_decay["params"]}
    norm_ids = {id(m.weight) for m in model.modules() if m.__class__.__name__ == "RMSNorm"}
    assert norm_ids <= {id(p) for p in no_decay["params"]}
    assert opt.defaults["betas"] == (0.9, 0.95)


# ---------------------------------------------------------------- datos
def test_sampler_shapes_and_shift(bin_file):
    s = BatchSampler(bin_file, block_size=16, batch_size=4, seed=1)
    x, y = s.get_batch()
    assert x.shape == y.shape == (4, 16) and x.dtype == y.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_sampler_deterministic_and_restorable(bin_file):
    a = BatchSampler(bin_file, 16, 4, seed=3)
    b = BatchSampler(bin_file, 16, 4, seed=3)
    assert torch.equal(a.get_batch()[0], b.get_batch()[0])
    state = copy.deepcopy(a.state_dict())
    want = [a.get_batch()[0] for _ in range(3)]
    c = BatchSampler(bin_file, 16, 4, seed=99)
    c.load_state_dict(state)
    got = [c.get_batch()[0] for _ in range(3)]
    assert all(torch.equal(w, g) for w, g in zip(want, got))


# ---------------------------------------------------------------- checkpoints
def make_state(bin_file, seed=0):
    torch.manual_seed(seed)
    model = GPT(TINY)
    opt = build_optimizer(model, tcfg())
    sampler = BatchSampler(bin_file, TINY.block_size, 2, seed=seed)
    return model, opt, sampler


def test_checkpoint_roundtrip(bin_file, tmp_path):
    model, opt, sampler = make_state(bin_file)
    c = tcfg()
    for step in range(3):
        run_step(model, opt, sampler, c, step)
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, opt, sampler, step=3, best_val=2.5)
    assert not list(tmp_path.glob("*.tmp"))
    m2, o2, s2 = make_state(bin_file, seed=42)
    meta = load_checkpoint(path, m2, o2, s2)
    assert meta["step"] == 3 and meta["best_val"] == 2.5
    for (n, p), (_, q) in zip(model.named_parameters(), m2.named_parameters()):
        assert torch.equal(p, q), n
    assert torch.equal(sampler.get_batch()[0], s2.get_batch()[0])


def test_rotation_keeps_last_and_best(tmp_path):
    for step in (50, 100, 150, 200):
        (tmp_path / f"ckpt-{step:07d}.pt").write_bytes(b"x")
    (tmp_path / "best.pt").write_bytes(b"x")
    rotate_checkpoints(tmp_path, keep=2)
    left = sorted(p.name for p in tmp_path.glob("*.pt"))
    assert left == ["best.pt", "ckpt-0000150.pt", "ckpt-0000200.pt"]


# ---------------------------------------------------------------- bucle
def test_exact_resume_is_bit_identical(bin_file, tmp_path):
    c = tcfg()
    m1, o1, s1 = make_state(bin_file)
    losses1 = [run_step(m1, o1, s1, c, step)[0] for step in range(6)]

    m2, o2, s2 = make_state(bin_file)
    losses2 = [run_step(m2, o2, s2, c, step)[0] for step in range(3)]
    save_checkpoint(tmp_path / "ck.pt", m2, o2, s2, step=3, best_val=9.0)
    m3, o3, s3 = make_state(bin_file, seed=77)              # estado inicial distinto a proposito
    meta = load_checkpoint(tmp_path / "ck.pt", m3, o3, s3)
    losses2 += [run_step(m3, o3, s3, c, step)[0] for step in range(meta["step"], 6)]

    assert losses1 == losses2
    for (n, p), (_, q) in zip(m1.named_parameters(), m3.named_parameters()):
        assert torch.equal(p, q), n


def test_accumulation_equals_big_batch():
    torch.manual_seed(0)
    model = GPT(TINY)
    x = torch.randint(0, 128, (8, 16))
    y = torch.randint(0, 128, (8, 16))
    model.zero_grad()
    _, loss = model(x, targets=y)
    loss.backward()
    big = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    mean_loss = accumulate_gradients(model, [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)])
    for g, p in zip(big, model.parameters()):
        assert torch.allclose(g, p.grad, atol=1e-6)
    assert mean_loss == pytest.approx(loss.item(), abs=1e-5)


def test_run_step_reports_grad_norm_and_sets_lr(bin_file):
    model, opt, sampler = make_state(bin_file)
    c = tcfg()
    loss, gnorm = run_step(model, opt, sampler, c, step=0)
    assert math.isfinite(loss) and math.isfinite(gnorm) and gnorm > 0
    assert opt.param_groups[0]["lr"] == pytest.approx(get_lr(0, c))
