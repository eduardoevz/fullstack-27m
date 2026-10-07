"""Generacion de texto: muestreo (temperatura, top-k, top-p) con cache KV.

    python src/sample.py --prompt "export default function Nav(" --lang ts
    python src/sample.py --prompt "from fastapi import FastAPI" --lang py --temperature 0.6 --seed 1

Los documentos de entrenamiento tienen la forma  <|file|> <|lang_x|> codigo <|endoftext|>,
asi que el prompt se enmarca igual y la generacion para al emitir <|endoftext|>.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def apply_repetition_penalty(logits: torch.Tensor, recent_ids: list[int], penalty: float) -> torch.Tensor:
    """Penaliza los tokens de `recent_ids` (estilo CTRL): los logits positivos se dividen y los negativos se multiplican."""
    logits = logits.clone()
    if penalty == 1.0 or not recent_ids:
        return logits
    idx = torch.tensor(sorted(set(recent_ids)))
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def banned_by_ngram(ids: list[int], n: int) -> set[int]:
    """Tokens que, si se emiten ahora, repetirian un n-grama que ya aparece en `ids`."""
    if n <= 0 or len(ids) < n - 1:
        return set()
    tail = tuple(ids[len(ids) - (n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n - 1]) == tail:
            banned.add(ids[i + n - 1])
    return banned


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float, min_p: float = 0.0) -> torch.Tensor:
    """Pone -inf a los tokens fuera de top-k, del nucleo top-p o por debajo de min_p x la prob. maxima."""
    logits = logits.clone()
    if min_p > 0.0:
        probs = torch.softmax(logits, dim=-1)
        logits[probs < min_p * probs.max()] = float("-inf")
    if top_k and top_k < logits.numel():
        logits[logits < torch.topk(logits, top_k).values[-1]] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cum - torch.softmax(sorted_logits, dim=-1) >= top_p     # el token que cruza p se conserva
        logits[order[drop]] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                generator: torch.Generator | None = None, min_p: float = 0.0) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(filter_logits(logits / temperature, top_k, top_p, min_p), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.8,
             top_k: int = 40, top_p: float = 0.95, eos_id: int | None = None,
             generator: torch.Generator | None = None, on_token=None,
             repetition_penalty: float = 1.0, repetition_window: int = 64,
             no_repeat_ngram_size: int = 0, min_p: float = 0.0) -> list[int]:
    """Devuelve solo los tokens nuevos (sin el eos). Si el contexto llena `block_size`, descarta la
    cache y vuelve a procesar la mitad final de la ventana (la posicion es relativa gracias a RoPE).
    `on_token(id)` se llama con cada token nuevo (para mostrar el texto a medida que sale).
    Anti-repeticion (apagada por defecto): `repetition_penalty` y `no_repeat_ngram_size` miran solo los tokens
    GENERADOS (no el prompt: el codigo repite con razon los identificadores que ya escribio el usuario);
    la penalizacion usa los ultimos `repetition_window`. `min_p` filtra tokens con prob. < min_p x la maxima."""
    if not prompt_ids:
        raise ValueError("el prompt no puede estar vacio")
    window = model.cfg.block_size
    ids = list(prompt_ids)
    logits, cache = model.forward_cached(torch.tensor([ids[-(window - 1):]]), None)
    out: list[int] = []
    for _ in range(max_new_tokens):
        step_logits = logits[0, -1]
        if repetition_penalty != 1.0:
            step_logits = apply_repetition_penalty(step_logits, out[-repetition_window:], repetition_penalty)
        if no_repeat_ngram_size > 0:
            banned = banned_by_ngram(out, no_repeat_ngram_size)
            if banned:
                step_logits = step_logits.clone()
                step_logits[list(banned)] = float("-inf")
        nxt = sample_next(step_logits, temperature, top_k, top_p, generator, min_p)
        if nxt == eos_id:
            break
        out.append(nxt)
        ids.append(nxt)
        if on_token:
            on_token(nxt)
        if cache[0][0].shape[2] + 1 > window:
            logits, cache = model.forward_cached(torch.tensor([ids[-(window // 2):]]), None)
        else:
            logits, cache = model.forward_cached(torch.tensor([[nxt]]), cache)
    return out


def load_model(ckpt_path: str | Path, cfg):
    """Carga solo los pesos del modelo desde un checkpoint de entrenamiento."""
    from src.model.gpt import GPT
    model = GPT(cfg.model)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval(), state.get("step")


def frame_prompt(tok, text: str, lang: str) -> list[int]:
    """<|file|> <|lang_x|> + texto. El texto se codifica sin tokens finales."""
    from src.tokenizer.pretokenizer import lang_token
    head = [tok.token_to_id("<|file|>"), tok.token_to_id(lang_token(lang))]
    return head + tok.encode(text).ids


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None, help="por defecto best.pt de la carpeta de checkpoints")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--lang", default="ts", choices=["js", "jsx", "ts", "tsx", "py"])
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--rep-penalty", type=float, default=1.1, help="1.1 por defecto (Fase 8); 1.0 = como v1")
    ap.add_argument("--no-repeat-ngram", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    new = generate(model, frame_prompt(tok, args.prompt, args.lang), args.max_new_tokens,
                   args.temperature, args.top_k, args.top_p, tok.token_to_id("<|endoftext|>"), gen,
                   repetition_penalty=args.rep_penalty, no_repeat_ngram_size=args.no_repeat_ngram, min_p=args.min_p)
    sys.stdout.reconfigure(encoding="utf-8")
    print(args.prompt + tok.decode(new, skip_special_tokens=False))


if __name__ == "__main__":
    main()
