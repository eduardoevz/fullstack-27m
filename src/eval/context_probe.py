"""Fase 12, medicion previa: ¿el contexto de 512 tokens limita al modelo? ¿Next.js y FastAPI siguen flojos?

    python src/eval/context_probe.py --config config/model_v2_long_c.json --ckpt <hito-0018310.pt> --out benchmarks/v2_context_probe.json

Reglas fijadas ANTES de medir:
  Contexto. Se toman documentos de val.bin que miden >= 513 tokens y se mide la perdida de los 64 tokens que ocupan las posiciones
  449-512 del documento, alimentando al modelo solo con los ultimos c tokens previos (c = 64, 128, 256, 384, 512). Son los mismos
  objetivos para todo c, asi que la diferencia es solo contexto. Se llama ganancia(a->b) = perdida(a) - perdida(b).
    limita     si ganancia(256->512) >= 0,05  y  ganancia(384->512) >= 0,015  y  >= 25 % de los tokens de val estan en la posicion >= 512 de su documento
    no limita  si ganancia(384->512) < 0,008
    dudoso     en otro caso
  Dominio. flojo si la perdida del dominio (val_<dominio>.bin) supera en mas de un 10 % a la de la mezcla de validacion.
  Control (anadido tras ver que, con la ventana de 512 tokens, la perdida con contexto 512 salio PEOR que con 384: esa ventana incluye el inicio
  del documento y las otras no). Con `--position 768 --skip-domains` los objetivos son los tokens 705-768 y ninguna ventana (<= 512) toca el inicio del
  documento; la regla de clasificacion es la misma. Se informa como control, no sustituye al veredicto de la medicion principal.
Este script solo mide: no entrena ni decide la Fase 12.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.eval.diagnose import doc_spans
from src.eval.domain_loss import stream_loss

CONTEXTS = (64, 128, 256, 384, 512)
POSITION = 512                      # el objetivo termina en la posicion 512 del documento
TARGETS = 64                        # tokens objetivo (posiciones 449-512)
GAIN_FAR, GAIN_NEAR, SHARE_LONG, GAIN_NONE = 0.05, 0.015, 0.25, 0.008
DOMAIN_MARGIN = 1.10


def classify_context(loss_by_c: dict[int, float], share_long: float) -> str:
    far, near = loss_by_c[256] - loss_by_c[512], loss_by_c[384] - loss_by_c[512]
    if far >= GAIN_FAR and near >= GAIN_NEAR and share_long >= SHARE_LONG:
        return "limita"
    if near < GAIN_NONE:
        return "no limita"
    return "dudoso"


def classify_domain(domain_loss: float, mix_loss: float) -> str:
    return "flojo" if domain_loss > DOMAIN_MARGIN * mix_loss else "ok"


def target_window(data: np.ndarray, start: int, context: int, position: int = POSITION) -> np.ndarray:
    """Fila de `context` tokens de entrada + 1 de objetivo que termina en la posicion `position` del documento que empieza en `start`."""
    end = start + position
    return np.asarray(data[end - context:end + 1], dtype=np.int64)


def share_beyond(doc_lengths: list[int], position: int) -> float:
    """Fraccion de los tokens que estan en una posicion >= `position` de su documento."""
    total = sum(doc_lengths)
    return sum(max(0, n - position) for n in doc_lengths) / total if total else 0.0


@torch.no_grad()
def tail_loss(model, rows: np.ndarray, batch: int = 16) -> float:
    """Perdida media de los ultimos TARGETS tokens objetivo de cada fila."""
    total, count = 0.0, 0
    for i in range(0, len(rows), batch):
        t = torch.from_numpy(rows[i:i + batch])
        logits, _ = model(t[:, :-1])
        loss = F.cross_entropy(logits[:, -TARGETS:].transpose(1, 2), t[:, 1:][:, -TARGETS:], reduction="sum")
        total += float(loss)
        count += t.shape[0] * TARGETS
    return total / max(1, count)


def main() -> None:
    from tokenizers import Tokenizer

    from src.sample import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--docs", type=int, default=400)
    ap.add_argument("--max-windows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--position", type=int, default=POSITION, help="posicion del documento donde terminan los objetivos (512 = medicion principal)")
    ap.add_argument("--skip-domains", action="store_true", help="omite la perdida por dominio (para el control)")
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "v2_context_probe.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    file_id, eot_id = tok.token_to_id("<|file|>"), tok.token_to_id("<|endoftext|>")
    val_path = Path(cfg.paths.token_dir) / "val.bin"
    data = np.fromfile(val_path, dtype=np.uint16)
    spans = doc_spans(data, file_id, eot_id)
    lengths = [e - s + 1 for s, e in spans]
    long_docs = [s for (s, e) in spans if e - s + 1 >= args.position + 1]
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(long_docs, size=min(args.docs, len(long_docs)), replace=False)
    share_long = share_beyond(lengths, POSITION)

    loss_by_c = {c: tail_loss(model, np.stack([target_window(data, int(s), c, args.position) for s in pick])) for c in CONTEXTS}
    verdict = classify_context(loss_by_c, share_long)
    print(f"documentos de val: {len(spans)}; con >= {args.position + 1} tokens: {len(long_docs)}; "
          f"tokens en posicion >= {POSITION} de su documento: {share_long:.1%}")
    for c, v in loss_by_c.items():
        print(f"  contexto {c:>3}: perdida de los tokens {args.position - TARGETS + 1}-{args.position} = {v:.4f}")
    print(f"ganancia 256->512 = {loss_by_c[256] - loss_by_c[512]:.4f}; 384->512 = {loss_by_c[384] - loss_by_c[512]:.4f}  =>  contexto: {verdict}")

    block = cfg.model.block_size
    domains = {"mezcla de validacion": data}
    for p in ([] if args.skip_domains else sorted(Path(cfg.paths.token_dir).glob("val_*.bin"))):
        domains[p.stem.replace("val_", "dominio:")] = np.fromfile(p, dtype=np.uint16)
    dom = {}
    for name, d in domains.items():
        dom[name] = {**stream_loss(model, d, block=block, max_windows=args.max_windows), "file_tokens": int(len(d))}
    mix = dom["mezcla de validacion"]["loss"] if dom else 0.0
    for name, r in dom.items():
        if name != "mezcla de validacion":
            r["veredicto"] = classify_domain(r["loss"], mix)
        print(f"  {name:<24} perdida {r['loss']:.4f}  ({r['windows']} ventanas)  {r.get('veredicto', '')}")

    result = {"checkpoint_step": step, "position": args.position, "docs": int(len(pick)), "targets_per_doc": TARGETS, "share_tokens_beyond_512": share_long,
              "loss_by_context": loss_by_c, "gain_256_512": loss_by_c[256] - loss_by_c[512],
              "gain_384_512": loss_by_c[384] - loss_by_c[512], "contexto": verdict, "dominios": dom,
              "reglas": {"gain_far": GAIN_FAR, "gain_near": GAIN_NEAR, "share_long": SHARE_LONG, "gain_none": GAIN_NONE,
                         "domain_margin": DOMAIN_MARGIN}}
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
