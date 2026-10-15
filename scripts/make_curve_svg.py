"""Render the MT-Bench cost-quality curve from a committed mtbench_*.json as
light and dark SVGs for the README.

The circled point is the best in-sample point on the sweep; the reported headline
is the held-out one (see mtbench_holdout.py), which is why the label says so.

  PYTHONPATH=. python scripts/make_curve_svg.py data/mtbench_gpt4o.json \
      --pair "free Ministral-8B -> GPT-4o"
"""
from __future__ import annotations

import argparse
import json

W, H = 860, 430
L, R, T, B = 70, 24, 64, 54     # plot margins
BLUE = "#4269d0"
THEMES = {
    "light": dict(ink="#1f2328", ink2="#57606a", grid="#d8dee4", axis="#8c959f"),
    "dark": dict(ink="#e6edf3", ink2="#8b949e", grid="#30363d", axis="#6e7681"),
}


def render(points: list[tuple[float, float]], headline: tuple[float, float],
           pair: str, theme: dict) -> str:
    xs = [20, 102]
    ys = [q for _, q in points] + [headline[1], 95]
    y0, y1 = min(94, int(min(ys)) - 1), max(102, int(max(ys)) + 1)

    def X(v):
        return L + (v - xs[0]) / (xs[1] - xs[0]) * (W - L - R)

    def Y(v):
        return T + (y1 - v) / (y1 - y0) * (H - T - B)

    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="-apple-system,Segoe UI,Helvetica,Arial,sans-serif">',
         f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="{theme["ink"]}">'
         f'Triage on MT-Bench: cost reduction against quality</text>',
         f'<text x="{L}" y="42" font-size="12" fill="{theme["ink2"]}">{pair} · 80 questions · '
         f'LLM-judge · in-sample threshold sweep (headline reported on a held-out split)</text>']
    for v in range(y0, y1 + 1, 2):
        s.append(f'<line x1="{L}" y1="{Y(v)}" x2="{W - R}" y2="{Y(v)}" '
                 f'stroke="{theme["grid"]}" stroke-width="1"/>')
        s.append(f'<text x="{L - 8}" y="{Y(v) + 4}" font-size="11" text-anchor="end" '
                 f'fill="{theme["ink2"]}">{v}%</text>')
    for v in range(xs[0], xs[1], 10):
        s.append(f'<text x="{X(v)}" y="{H - B + 18}" font-size="11" text-anchor="middle" '
                 f'fill="{theme["ink2"]}">{v}%</text>')
    s.append(f'<line x1="{L}" y1="{Y(y0)}" x2="{W - R}" y2="{Y(y0)}" '
             f'stroke="{theme["axis"]}" stroke-width="1"/>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 14}" font-size="12" text-anchor="middle" '
             f'fill="{theme["ink2"]}">cost reduction vs always-paid →</text>')
    mid = (T + H - B) / 2
    s.append(f'<text x="16" y="{mid}" font-size="12" text-anchor="middle" '
             f'fill="{theme["ink2"]}" transform="rotate(-90 16 {mid})">quality vs paid model →</text>')
    s.append(f'<line x1="{L}" y1="{Y(95)}" x2="{W - R}" y2="{Y(95)}" stroke="{theme["axis"]}" '
             f'stroke-width="1" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{L + 6}" y="{Y(95) - 6}" font-size="11" fill="{theme["ink2"]}">'
             f'95% quality line</text>')
    path = " ".join(f'{"M" if i == 0 else "L"} {X(x):.1f} {Y(y):.1f}'
                    for i, (x, y) in enumerate(points))
    s.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2" '
             f'stroke-linejoin="round"/>')
    for x, y in points:
        s.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{BLUE}"/>')
    hx, hy = X(headline[0]), Y(headline[1])
    s.append(f'<circle cx="{hx}" cy="{hy}" r="6.5" fill="none" stroke="{BLUE}" stroke-width="2"/>')
    s.append(f'<text x="{hx - 10}" y="{hy - 12}" font-size="11.5" font-weight="600" '
             f'text-anchor="end" fill="{theme["ink"]}">'
             f'{headline[0]}% saved · {headline[1]}% quality (in-sample)</text>')
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default="data/mtbench_gpt4o.json")
    ap.add_argument("--pair", help="label for the model pair (default: from the JSON)")
    ap.add_argument("--out-dir", default="docs/assets")
    a = ap.parse_args()

    d = json.load(open(a.source))
    pair = a.pair or f"free {d['small']} → {d['big']}"
    points = sorted({(p["cost_reduction_pct"], p["quality_vs_big_pct"]) for p in d["curve"]})
    # Best in-sample point that holds >= 95% quality with a real escalation rate.
    best = max((p for p in d["curve"]
                if p["quality_vs_big_pct"] >= 95 and p["escalation_rate"] > 0),
               key=lambda p: p["cost_reduction_pct"])
    headline = (best["cost_reduction_pct"], best["quality_vs_big_pct"])

    for mode, theme in THEMES.items():
        out = f"{a.out_dir}/mtbench_curve_{mode}.svg"
        with open(out, "w", encoding="utf-8") as f:
            f.write(render(points, headline, pair, theme))
        print(f"wrote {out}  headline={headline}")


if __name__ == "__main__":
    main()
