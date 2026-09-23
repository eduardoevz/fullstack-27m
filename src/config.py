"""Carga y validacion de la configuracion del proyecto.

Un unico archivo JSON es la fuente de verdad para arquitectura, entrenamiento,
datos y rutas. Todo script del proyecto lee desde aqui; ningun hiperparametro
se escribe a mano en otro sitio.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "model_27m.json"


def _build(cls, data: dict):
    """Instancia una dataclass ignorando claves desconocidas y avisando de ellas."""
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: claves desconocidas en el config: {sorted(unknown)}")
    missing = known - set(data)
    if missing:
        raise ValueError(f"{cls.__name__}: faltan claves en el config: {sorted(missing)}")
    return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    block_size: int
    rope_theta: float
    norm_eps: float
    tie_embeddings: bool
    init_std: float

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError(f"d_model ({self.d_model}) debe ser divisible por n_head ({self.n_head})")
        if self.head_dim % 2:
            raise ValueError(f"head_dim ({self.head_dim}) debe ser par para RoPE")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    def param_count(self) -> int:
        """Conteo analitico de parametros. Los tests comparan el modelo real contra esto."""
        d, ff, v = self.d_model, self.d_ff, self.vocab_size
        attn = 4 * d * d              # qkv (3) + proyeccion de salida (1), sin bias
        mlp = 3 * d * ff              # SwiGLU: gate + up + down
        norms = 2 * d                 # dos RMSNorm por bloque
        per_block = attn + mlp + norms
        embeddings = v * d if self.tie_embeddings else 2 * v * d
        return self.n_layer * per_block + embeddings + d  # + RMSNorm final


@dataclass(frozen=True)
class TrainingConfig:
    micro_batch_size: int
    grad_accum_steps: int
    max_steps: int
    learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    dtype: str
    num_threads: int
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    sample_interval: int
    log_interval: int
    keep_last_checkpoints: int

    def tokens_per_step(self, block_size: int) -> int:
        return self.micro_batch_size * self.grad_accum_steps * block_size


@dataclass(frozen=True)
class DataConfig:
    target_train_tokens: int
    target_val_tokens: int
    languages: list
    min_bytes: int
    max_bytes: int
    min_alnum_ratio: float
    max_mean_line_length: int
    dedup_threshold: float


@dataclass(frozen=True)
class PathsConfig:
    data_root: str
    raw_dir: str
    token_dir: str
    checkpoint_dir: str
    tokenizer_file: str

    def ensure(self) -> None:
        for d in (self.data_root, self.raw_dir, self.token_dir, self.checkpoint_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    @property
    def train_bin(self) -> Path:
        return Path(self.token_dir) / "train.bin"

    @property
    def val_bin(self) -> Path:
        return Path(self.token_dir) / "val.bin"


@dataclass(frozen=True)
class Config:
    name: str
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    paths: PathsConfig
    source: Path


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Config(
        name=raw["name"],
        model=_build(ModelConfig, raw["model"]),
        training=_build(TrainingConfig, raw["training"]),
        data=_build(DataConfig, raw["data"]),
        paths=_build(PathsConfig, raw["paths"]),
        source=path.resolve(),
    )
