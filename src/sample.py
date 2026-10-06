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


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Pone -inf a los tokens fuera de top-k y fuera del nucleo de probabilidad acumulada top-p."""
    logits = logits.clone()
    if top_k and top_k < logits.numel():
        logits[logits < torch.topk(logits, top_k).values[-1]] = float("-inf")
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        drop = cum - torch.softmax(sorted_logits, dim=-1) >= top_p     # el token que cruza p se conserva
        logits[order[drop]] = float("-inf")
    return logits


def sample_next(logits: torch.Tensor, temperature: float, top_k: int, top_p: float,
                generator: torch.Generator | None = None) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))


@torch.no_grad()
def generate(model, prompt_ids: list[int], max_new_tokens: int, temperature: float = 0.8,
             top_k: int = 40, top_p: float = 0.95, eos_id: int | None = None,
             generator: torch.Generator | None = None, on_token=None) -> list[int]:
    """Devuelve solo los tokens nuevos (sin el eos). Si el contexto llena `block_size`, descarta la
    cache y vuelve a procesar la mitad final de la ventana (la posicion es relativa gracias a RoPE).
    `on_token(id)` se llama con cada token nuevo (para mostrar el texto a medida que sale)."""
    if not prompt_ids:
        raise ValueError("el prompt no puede estar vacio")
    window = model.cfg.block_size
    ids = list(prompt_ids)
    logits, cache = model.forward_cached(torch.tensor([ids[-(window - 1):]]), None)
    out: list[int] = []
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[0, -1], temperature, top_k, top_p, generator)
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
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.set_num_threads(cfg.training.num_threads)
    tok = Tokenizer.from_file(cfg.paths.tokenizer_file)
    model, step = load_model(args.ckpt or Path(cfg.paths.checkpoint_dir) / "best.pt", cfg)
    gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
    new = generate(model, frame_prompt(tok, args.prompt, args.lang), args.max_new_tokens,
                   args.temperature, args.top_k, args.top_p, tok.token_to_id("<|endoftext|>"), gen)
    sys.stdout.reconfigure(encoding="utf-8")
    print(args.prompt + tok.decode(new, skip_special_tokens=False))


if __name__ == "__main__":
    main()
