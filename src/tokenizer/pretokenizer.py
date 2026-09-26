"""Pre-tokenizer y tokens especiales del BPE orientado a codigo.

El regex decide *donde no se puede fusionar*: BPE nunca cruza un limite de
pre-token. Diseno:
  1. sangria al inicio de linea (`\n` + espacios/tabs) como una unidad, para que
     BPE aprenda por si solo tokens de 2/4/8 espacios y tab;
  2. operadores compuestos (`===`, `=>`, `?.`, `::`...) como unidad;
  3. identificadores completos (camelCase / snake_case no se parten a la fuerza);
  4. numeros, simbolos sueltos y espacios interiores.
Un espacio previo se pega a la palabra/operador siguiente (` const`, ` ===`).
"""

from __future__ import annotations

from tokenizers import Regex, pre_tokenizers

SPECIAL_TOKENS = [
    "<|endoftext|>", "<|file|>", "<|lang_js|>", "<|lang_ts|>", "<|lang_py|>", "<|pad|>",
]
_LANG_TOKEN = {"js": "<|lang_js|>", "jsx": "<|lang_js|>", "ts": "<|lang_ts|>", "tsx": "<|lang_ts|>", "py": "<|lang_py|>"}


def lang_token(lang: str) -> str:
    return _LANG_TOKEN[lang]


OPERATORS = r"===|!==|\*\*=|\.\.\.|<<=|>>=|=>|==|!=|<=|>=|&&|\|\||\?\?|\?\.|::|\*\*|->|<<|>>|\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^="

PATTERN = "|".join([
    r"\r?\n[ \t]*",                                   # sangria como unidad
    rf" ?(?:{OPERATORS})",                            # operadores compuestos
    r" ?[\p{L}_$][\p{L}\p{N}_$]*",                    # identificadores
    r" ?\p{N}+",                                      # numeros
    r" ?[^\s\p{L}\p{N}_$]",                           # simbolo suelto
    r"[ \t]+(?=[ \t])",                               # tramos de espacios interiores
    r"[ \t]",                                         # espacio suelto
    r"\s",                                            # cualquier otro blanco
])


def build_pretokenizer() -> pre_tokenizers.PreTokenizer:
    return pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
