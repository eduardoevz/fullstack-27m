"""Vigilancia de FIM durante el entrenamiento largo (Fase 11) con aborto temprano.

Cada `fim_monitor_interval` pasos se mide, sobre casos fijos de validacion con el medio de lineas completas:
  - perdida teacher-forced del medio (prefijo+sufijo) frente a la de v1 solo con prefijo (referencia fija);
  - tasa de parada: % de generaciones (T=0) que emiten <|endoftext|> antes de agotar el presupuesto.
La regla de aborto se escribe en el config ANTES de entrenar (`fim_abort_*`); si falla, el entrenamiento guarda y sale.

    python -m src.eval.fim_monitor --ref        # calcula la referencia de v1 (una vez) y la guarda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.eval.fim_eval import MAX_PREFIX_TOKENS, MAX_SUFFIX_TOKENS, build_cases, loss_on_middle, trim_context


def newline_token_ids(tok) -> list:
    """Ids de los tokens cuyo texto contiene un salto de linea (los cortes por linea caen justo despues)."""
    vocab = tok.get_vocab()
    return sorted(i for i in vocab.values() if "\n" in tok.decode([i], skip_special_tokens=False))


def prepare_cases(tok, token_dir: Path, n: int, seed: int = 0) -> list:
    """`n` casos fijos de infilling con contexto ya recortado: [{lang, prefix, suffix, middle}]."""
    from src.eval.fim_eval import load_docs
    docs = load_docs(tok, token_dir, per_file=max(1, n // 5 + 20), seed=seed)
    count = lambda s: len(tok.encode(s).ids)
    out = []
    for c in build_cases(docs, n=n, seed=seed):
        pre, suf = trim_context(c["prefix_full"], c["suffix_full"], count, MAX_PREFIX_TOKENS, MAX_SUFFIX_TOKENS)
        out.append({"lang": c["lang"], "prefix": pre, "suffix": suf, "middle": c["middle"]})
    return out


@torch.no_grad()
def run_monitor(model, tok, cases: list, fim: bool = True) -> dict:
    """{loss_mid, stop_rate, n}. `fim=False` = prefijo solo (v1): la tasa de parada no aplica (None)."""
    from src.sample import frame_fim_prompt, frame_prompt, generate
    eot = tok.token_to_id("<|endoftext|>")
    was_training = model.training
    model.eval()
    losses, stops = [], []
    try:
        for c in cases:
            mid_ids = tok.encode(c["middle"]).ids
            prompt = (frame_fim_prompt(tok, c["prefix"], c["suffix"], c["lang"]) if fim
                      else frame_prompt(tok, c["prefix"], c["lang"]))
            r = loss_on_middle(model, prompt, mid_ids)
            if r is not None:
                losses.append(r[0])
            if fim:
                budget = min(80, 2 * len(mid_ids) + 8)
                stops.append(float(len(generate(model, prompt, budget, 0.0, eos_id=eot)) < budget))
    finally:
        model.train(was_training)
    return {"loss_mid": float(np.mean(losses)) if losses else None,
            "stop_rate": float(np.mean(stops)) if stops else None, "n": len(losses)}


def decide_abort(metrics: dict, ref_loss: float, step: int, abort_step: int, max_ratio: float, min_stop: float):
    """None = seguir; str = motivo del aborto. Solo se evalua desde `abort_step` (0 = nunca abortar)."""
    if abort_step <= 0 or step < abort_step:
        return None
    reasons = []
    ratio = metrics["loss_mid"] / ref_loss
    if ratio > max_ratio:
        reasons.append(f"perdida del medio {metrics['loss_mid']:.3f} = {ratio:.3f} x la de v1 (limite {max_ratio})")
    if metrics["stop_rate"] < min_stop:
        reasons.append(f"tasa de parada {metrics['stop_rate']:.2f} < {min_stop}")
    return "; ".join(reasons) or None


def main() -> None:
    from tokenizers import Tokenizer

    from src.config import REPO_ROOT, load_config
    from src.sample import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "model_v2.json"))
    ap.add_argument("--ckpt-v1", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "fim_monitor_ref.json"))
    args = ap.parse_args()
    cfg1, cfg2 = load_config(), load_config(args.config)
    torch.set_num_threads(cfg2.training.num_threads)
    tok = Tokenizer.from_file(cfg2.paths.tokenizer_file)
    cases = prepare_cases(tok, Path(cfg2.paths.token_dir), args.n)
    v1, _ = load_model(args.ckpt_v1 or Path(cfg1.paths.checkpoint_dir) / "v1-best.pt", cfg1)
    res = run_monitor(v1, tok, cases, fim=False)
    res["cases"] = len(cases)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
