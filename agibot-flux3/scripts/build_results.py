"""results/*/summary.json -> results/RESULTS.md (one table per split: model x k -> metrics).

    python scripts/build_results.py --results results --out results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/RESULTS.md")
    args = p.parse_args()
    root = Path(args.results)
    models = {d.name: json.loads((d / "summary.json").read_text()) for d in sorted(root.iterdir())
              if (d / "summary.json").exists()}
    splits = sorted({r["split"] for s in models.values() for r in s.values()})
    lines = ["# Results", "", "Fixed windows per split (seeded), Euler 30 steps, CFG 4, shift 5; metrics against the",
             "ground-truth future frames (PSNR/SSIM higher is better, LPIPS/FVD lower); judge = yes-rate with the true",
             "caption minus yes-rate with a swapped caption (Qwen3-VL).", ""]
    for split in splits:
        lines += [f"## {split}", "", "| model | k | PSNR | SSIM | LPIPS | FVD (n) | judge adherence | n_recon |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, summ in models.items():
            for key in sorted(summ, key=lambda s: summ[s]["k"]):
                r = summ[key]
                if r["split"] != split:
                    continue
                fvd = f"{r['fvd']:.1f} ({r.get('n_fvd')})" if "fvd" in r else "-"
                judge = fmt(r["judge"]["adherence"]) if "judge" in r else "-"
                lines.append(f"| {name} | {r['k']} | {fmt(r['psnr'], 2)} | {fmt(r['ssim'])} | {fmt(r['lpips'])} | {fvd} | "
                             f"{judge} | {r['n_recon']} |")
        lines.append("")
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}: {len(models)} models, splits {splits}")


if __name__ == "__main__":
    main()
