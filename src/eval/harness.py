"""Piezas reutilizables de la evaluacion: decodificacion, generacion de fragmentos, puntuacion y resumen con IC."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from src.eval.stats import bootstrap_ci
from src.eval.syntax import (RETENTION_MIN, check_fragments, has_undefined_names, plan_counts, repeated_line_ratio,
                             trim_to_valid)
from src.sample import frame_prompt, generate

FRAMING = {"js": "js", "jsx": "js", "ts": "ts", "tsx": "ts", "py": "py"}


@dataclass(frozen=True)
class Decoding:
    """Parametros de muestreo. Los valores por defecto son los de la v1 (sin anti-repeticion)."""
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    repetition_window: int = 64
    no_repeat_ngram_size: int = 0
    min_p: float = 0.0

    def generate_kwargs(self) -> dict:
        return asdict(self)

    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


def generate_fragments(model, tok, n: int, max_new_tokens: int, decoding: Decoding, seed: int = 0,
                       progress: bool = False) -> list[dict]:
    """n fragmentos desde cero (solo <|file|><|lang_x|>), repartidos 34/33/33 entre js, ts y py."""
    eot = tok.token_to_id("<|endoftext|>")
    head = {l: [tok.token_to_id("<|file|>"), tok.token_to_id(f"<|lang_{l}|>")] for l in ("js", "ts", "py")}
    rows, i = [], 0
    for lang, count in plan_counts(n).items():
        for _ in range(count):
            g = torch.Generator().manual_seed(seed + i)
            new = generate(model, head[lang], max_new_tokens, eos_id=eot, generator=g, **decoding.generate_kwargs())
            rows.append({"id": f"gen-{lang}-{i}", "lang": lang, "finished": len(new) < max_new_tokens,
                         "n_tokens": len(new), "code": tok.decode(new, skip_special_tokens=False)})
            i += 1
            if progress:
                print(f"\rgenerados {i}/{n}", end="", flush=True)
    if progress:
        print()
    return rows


def score(rows_in: list[dict]) -> list[dict]:
    """rows_in: [{id, lang, code, finished}] -> filas con ok / trimmed_ok / retention / repeat."""
    verdicts = check_fragments([{"id": r["id"], "lang": r["lang"], "code": r["code"]} for r in rows_in])
    rows = []
    for r, v in zip(rows_in, verdicts):
        if v["ok"]:
            trimmed_ok, retention = True, 1.0
        else:
            _, retention = trim_to_valid(r["code"], r["lang"])
            trimmed_ok = retention >= RETENTION_MIN
        rows.append({**r, "ok": v["ok"], "error": v["error"], "trimmed_ok": trimmed_ok, "retention": retention,
                     "repeat": repeated_line_ratio(r["code"]),
                     "undefined_names": has_undefined_names(r["code"]) if (r["lang"] == "py" and v["ok"]) else None})
    return rows


def run_prompt_set(model, tok, prompts: list[dict], samples: int, max_new_tokens: int, decoding: Decoding,
                   seed: int = 0) -> list[dict]:
    """`samples` generaciones por prompt. La muestra 0 usa la semilla `seed + i` (igual que la suite de la Fase 7)."""
    eot = tok.token_to_id("<|endoftext|>")
    out = []
    for j in range(samples):
        for i, p in enumerate(prompts):
            ids = frame_prompt(tok, p["prompt"], FRAMING[p["lang"]])
            g = torch.Generator().manual_seed(seed + 1000 * j + i)
            new = generate(model, ids, max_new_tokens, eos_id=eot, generator=g, **decoding.generate_kwargs())
            completion = tok.decode(new, skip_special_tokens=False)
            full = p["prompt"] + completion
            ok = check_fragments([{"id": 0, "lang": p["lang"], "code": full}])[0]["ok"]
            retention = 1.0 if ok else trim_to_valid(full, p["lang"])[1]
            out.append({"id": p["id"], "sample": j, "lang": p["lang"], "description": p["description"],
                        "prompt": p["prompt"], "completion": completion, "finished": len(new) < max_new_tokens,
                        "strict_ok": ok, "trimmed_ok": ok or retention >= RETENTION_MIN, "retention": retention,
                        "on_topic": any(e in completion for e in p["expect"]),
                        "repeat": repeated_line_ratio(completion)})
    return out


def summarize_prompts(rows: list[dict], seed: int = 0) -> dict:
    """Promedia las muestras dentro de cada prompt y calcula el IC95 % por bootstrap sobre prompts (la unidad independiente)."""
    by_id: dict[str, list[dict]] = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    out = {"n_prompts": len(by_id), "n_samples": len(rows)}
    for key in ("strict_ok", "trimmed_ok", "on_topic"):
        per_prompt = [sum(bool(r[key]) for r in rs) / len(rs) for rs in by_id.values()]
        mean, lo, hi = bootstrap_ci(per_prompt, seed=seed)
        out[key.replace("_ok", "")] = {"mean": mean, "lo": lo, "hi": hi}
    return out
