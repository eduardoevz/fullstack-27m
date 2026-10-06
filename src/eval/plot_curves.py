"""Dibuja las curvas de entrenamiento en un SVG autocontenido (sin matplotlib).

    python src/eval/plot_curves.py        # lee logs/train.csv y escribe docs/curvas-entrenamiento.svg

Pasos 0-15.260 en X; pérdida en Y. Train: media móvil de 20 puntos (cada punto = 10 pasos); validación: cada 500 pasos.
Se recorta el eje a 1,3-3,5 para que se aprecie la zona útil (los primeros pasos, con pérdida > 3,5, quedan fuera); el rango completo está en el CSV.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import REPO_ROOT

W, H, L, R, T, B = 760, 420, 62, 20, 36, 46
YMIN, YMAX = 1.3, 3.5


def smooth(xs, ys, k=20):
    out = []
    for i in range(len(ys)):
        lo = max(0, i - k + 1)
        out.append((xs[i], sum(ys[lo:i + 1]) / (i + 1 - lo)))
    return out


def main() -> None:
    rows = list(csv.DictReader(open(REPO_ROOT / "logs" / "train.csv", encoding="utf-8")))
    tr = [(int(r["step"]), float(r["loss"])) for r in rows if r["loss"]]
    va = [(int(r["step"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
    xmax = max(s for s, _ in tr)
    sx = lambda s: L + (W - L - R) * s / xmax
    sy = lambda v: T + (H - T - B) * (1 - (min(max(v, YMIN), YMAX) - YMIN) / (YMAX - YMIN))
    path = lambda pts: " ".join(f"{'M' if i == 0 else 'L'}{sx(s):.1f},{sy(v):.1f}" for i, (s, v) in enumerate(pts))

    trs = [p for p in smooth([s for s, _ in tr], [v for _, v in tr]) if p[1] <= YMAX]
    g = []
    for v in (1.5, 2.0, 2.5, 3.0, 3.5):
        g.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#8884" stroke-width="1"/>'
                 f'<text x="{L - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for s in range(0, 15001, 3000):
        g.append(f'<text x="{sx(s):.1f}" y="{H - B + 18}" text-anchor="middle">{s:,}</text>'.replace(",", "."))
    last_v = va[-1]
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif" font-size="12" fill="#888">
<title>Curvas de entrenamiento de fullstack-27m</title>
<text x="{L}" y="22" font-size="14" font-weight="600" fill="#666">Pérdida de entrenamiento y validación (fullstack-27m, 15.260 pasos)</text>
{''.join(g)}
<path d="{path(trs)}" fill="none" stroke="#4c8bf5" stroke-width="1.6"/>
<path d="{path(va)}" fill="none" stroke="#e8612c" stroke-width="2"/>
{''.join(f'<circle cx="{sx(s):.1f}" cy="{sy(v):.1f}" r="2.4" fill="#e8612c"/>' for s, v in va)}
<text x="{sx(last_v[0]) - 6:.1f}" y="{sy(last_v[1]) - 10:.1f}" text-anchor="end" fill="#e8612c" font-weight="600">val {last_v[1]:.3f}</text>
<text x="{W - R}" y="{H - 8}" text-anchor="end">paso de optimización</text>
<rect x="{W - 250}" y="{T + 6}" width="12" height="3" fill="#4c8bf5"/><text x="{W - 232}" y="{T + 11}">entrenamiento (media móvil)</text>
<rect x="{W - 250}" y="{T + 24}" width="12" height="3" fill="#e8612c"/><text x="{W - 232}" y="{T + 29}">validación (cada 500 pasos)</text>
</svg>
"""
    out = REPO_ROOT / "docs" / "curvas-entrenamiento.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"escrito {out} ({len(tr)} puntos de train, {len(va)} de validacion)")


if __name__ == "__main__":
    main()
