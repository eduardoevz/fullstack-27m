"""Carga solo los pesos de un checkpoint (p. ej. v1) en un modelo con mas filas de embedding.

Optimizador y paso empiezan de cero. Las filas nuevas (tokens FIM) se inicializan con la media de
las existentes, de modo que los logits de los ids viejos no cambian.
"""

from __future__ import annotations

from pathlib import Path

import torch


def load_v1_weights(model: torch.nn.Module, ckpt_path: str | Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    own = model.state_dict()
    if set(state) != set(own):
        raise ValueError(f"las claves del checkpoint no coinciden con el modelo: {sorted(set(state) ^ set(own))[:5]}")
    merged, old_vocab = {}, None
    for k, v in state.items():
        t = own[k]
        if v.shape == t.shape:
            merged[k] = v
        elif v.dim() == 2 and v.shape[1] == t.shape[1] and v.shape[0] < t.shape[0]:
            extra = v.mean(0, keepdim=True).expand(t.shape[0] - v.shape[0], -1)
            merged[k] = torch.cat([v, extra.to(v.dtype)])
            old_vocab = v.shape[0]
        else:
            raise ValueError(f"{k}: forma {tuple(v.shape)} del checkpoint incompatible con {tuple(t.shape)} del modelo")
    model.load_state_dict(merged)
    new_vocab = model.tok_emb.weight.shape[0]
    return {"step": ckpt.get("step"), "old_vocab": old_vocab or new_vocab, "new_vocab": new_vocab}
