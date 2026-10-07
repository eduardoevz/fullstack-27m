"""Ajuste de la decodificacion de v1 sobre el conjunto DEV (nunca sobre test). Fase 8, paso 6.

    python src/eval/tune_decoding.py --out benchmarks/decoding_search.json

Etapa A: un parametro por vez desde la base (11 configuraciones). Etapa B: combinaciones de las ganadoras.
Reglas fijadas ANTES de medir:
  objetivo = (estricta + (1 - fraccion de fragmentos repetitivos)) / 2        [syntax_check, n fragmentos a 256 tokens]
  admisible: «en tema» (suite DEV) no cae mas de 5 puntos frente a la base, y la repeticion media > 0 (sin colapso)
  se elige la mejor admisible SOLO si supera a la base en >= MIN_GAIN (0,05) de objetivo; si no, se conserva la base.
El registro completo (tambien las configuraciones malas) se guarda y se reanuda si se interrumpe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT, load_config
from src.eval.harness import Decoding, generate_fragments, run_prompt_set, score, summarize_prompts
from src.eval.prompts import DEV_PROMPTS
from src.eval.syntax import summarize
from src.sample import load_model

MIN_GAIN = 0.05
TOPIC_TOLERANCE = 0.05
FAMILIES = {"rep": ("repetition_penalty", [1.05, 1.1, 1.2]), "ngram": ("no_repeat_ngram_size", [3, 4, 6]),
            "minp": ("min_p", [0.05, 0.1]), "temp": ("temperature", [0.6, 0.7])}


def objective(strict: float, repetitive_share: float) -> float:
    return (strict + (1.0 - repetitive_share)) / 2.0


def admissible(base_on_topic: float, on_topic: float, mean_repeat: float) -> bool:
    return on_topic >= base_on_topic - TOPIC_TOLERANCE and mean_repeat > 0.0


def build_stage_a() -> list[tuple[str, Decoding]]:
    out = [("base", Decoding())]
    for fam, (field, values) in FAMILIES.items():
        for v in values:
            out.append((f"{fam}={v}", Decoding(**{field: v})))
    return out


def choose(rows: list[dict]) -> dict:
    base = next(r for r in rows if r["name"] == "base")
    ok = [r for r in rows if r["name"] == "base" or admissible(base["on_topic"], r["on_topic"], r["mean_repeat"])]
    best = max(ok, key=lambda r: r["objective"])
    return best if best["objective"] - base["objective"] >= MIN_GAIN else base


def build_stage_b(stage_a: list[dict]) -> list[tuple[str, Decoding]]:
    """Combina las ganadoras (la mejor de cada familia que supera a la base por el margen y es admisible)."""
    base = next(r for r in stage_a if r["name"] == "base")
    winners = []
    for fam in FAMILIES:
        cands = [r for r in stage_a if r.get("family") == fam
                 and admissible(base["on_topic"], r["on_topic"], r["mean_repeat"])
                 and r["objective"] - base["objective"] >= MIN_GAIN]
        if cands:
            winners.append(max(cands, key=lambda r: r["objective"]))
    winners.sort(key=lambda r: r["objective"], reverse=True)
    if len(winners) < 2:
        return []
    default = Decoding().generate_kwargs()

    def merge(ws: list[dict]) -> Decoding:
        kw = dict(default)
        for w in ws:
            kw.update({k: v for k, v in w["decoding"].items() if v != default[k]})
        return Decoding(**kw)

    groups = [winners] + [list(c) for n in (2, 3) for c in combinations(winners[:3], n)]
    out, seen = [], set()
    for g in groups:
        d = merge(g)
        if d not in seen:
            seen.add(d)
            out.append(("+".join(w["name"] for w in g), d))
    return out[:4]


def evaluate(model, tok, name: str, family: str, decoding: Decoding, n_frag: int, dev_samples: int, seed: int) -> dict:
    t0 = time.time()
    frags = score(generate_fragments(model, tok, n_frag, 256, decoding, seed))
    s = summarize(frags)["total"]
    dev = summarize_prompts(run_prompt_set(model, tok, DEV_PROMPTS, dev_samples, 200, decoding, seed), seed)
    return {"name": name, "family": family, "decoding": decoding.generate_kwargs(),
            "strict": s["strict"], "trimmed": s["trimmed"], "repetitive_share": s["repetitive_share"],
            "mean_repeat": s["mean_repeat"], "finished_n": s["finished_n"],
            "objective": objective(s["strict"], s["repetitive_share"]),
            "on_topic": dev["on_topic"]["mean"], "dev_strict": dev["strict"]["mean"], "dev_trimmed": dev["trimmed"]["mean"],
            "seconds": round(time.time() - t0)}


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-fragments", type=int, default=45)
    ap.add_argument("--dev-samples", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "benchmarks" / "decoding_search.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    ckpt = args.ckpt or Path(cfg.paths.checkpoint_dir) / "v1-best.pt"
    model, step = load_model(ckpt, cfg)

    out_path = Path(args.out)
    state = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"rows": []}
    done = {r["name"] for r in state["rows"]}

    def save():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")

    def run(name, family, d):
        if name in done:
            return
        r = evaluate(model, tok, name, family, d, args.n_fragments, args.dev_samples, args.seed)
        state["rows"].append(r)
        done.add(name)
        save()
        print(f"{name:<22} objetivo={r['objective']:.3f} estricta={r['strict']:.0%} repetitivos={r['repetitive_share']:.0%} "
              f"en_tema_dev={r['on_topic']:.0%} ({r['seconds']}s)", flush=True)

    state.update({"checkpoint": str(ckpt), "checkpoint_step": step, "n_fragments": args.n_fragments,
                  "dev_samples": args.dev_samples, "rules": {"min_gain": MIN_GAIN, "topic_tolerance": TOPIC_TOLERANCE}})
    for name, d in build_stage_a():
        run(name, "base" if name == "base" else name.split("=")[0], d)
    stage_b = build_stage_b(state["rows"])
    state["stage_b_names"] = [n for n, _ in stage_b]
    for name, d in stage_b:
        run(name, "combo", d)
    best = choose(state["rows"])
    state["chosen"] = best
    state["decision"] = ("sin mejora por decodificacion: se conserva la base" if best["name"] == "base"
                         else f"se elige {best['name']}")
    save()
    print("\nDecision:", state["decision"])


if __name__ == "__main__":
    main()
