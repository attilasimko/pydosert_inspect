"""Ablation of pencil_k51, built on the FFT engine with the 51-px kernel.

    baseline_k25   the original: direct convolution, 25-px kernel (+-24 mm), kernels
                   zeroed below 0.5 mm radiological depth
    fft_k51        FFT convolution, 51-px kernel (+-50 mm), same cutoff -- the base
                   of the rest
    abl_pencil     fft_k51 + each pencil's own radiological depth (PencilDepthEngine)
    abl_no_cutoff  fft_k51 + kernels no longer zeroed below 0.5 mm radiological depth
    abl_contam     fft_k51 + the machine's electron contamination, on the central-axis depth
    pencil_k51     fft_k51 + all three (PencilDepthEngine)

The variants live in engine_lab.VARIANTS, and the run is engine_lab's own: same
loading, dose, gamma and results.csv. engine_lab's summary pools the cohorts, so
per-cohort tables with paired Wilcoxon tests are added at the end: against
baseline_k25 (the total gain) and against fft_k51 (what each single change adds).
Single-change gains need not add up to pencil_k51's: the changes interact (the
contamination, for one, is attenuated with whichever depth the kernels use).

    python ablation.py --limit 3              # smoke test, GoldAtlas
    python ablation.py --cohort both          # any engine_lab option passes through
"""
import csv
import sys
from datetime import datetime

import numpy as np
from scipy.stats import wilcoxon

import engine_lab as L

ABLATION = ["baseline_k25", "fft_k51", "abl_pencil", "abl_no_cutoff", "abl_contam",
            "pencil_k51"]
REFS = ["baseline_k25", "fft_k51"]


def per_cohort(log, run: str, variants: list[str]) -> None:
    with open(log, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["run"] == run]
    for cohort in sorted({r["cohort"] for r in rows}):
        by = {(r["patient"], r["variant"]): r for r in rows if r["cohort"] == cohort}
        done = sorted(p for p in {p for p, _ in by} if all((p, v) in by for v in variants))
        if not done:
            continue
        for ref in (r for r in REFS if r in variants):
            print(f"\n=== {cohort}: {len(done)} patients, paired against {ref} "
                  f"(mean, gain, patients improved, Wilcoxon p)")
            print(f"  {'variant':<14}" + "".join(f"{k:>30}" for k in ("gamma", "shell", "deep"))
                  + f"{'target':>8}{'sec':>7}{'GB':>6}")
            for v in variants:
                line = f"  {v:<14}"
                for key in ("gamma", "gamma_shell", "gamma_deep"):
                    x = np.array([float(by[(p, v)][key]) for p in done])
                    d = x - np.array([float(by[(p, ref)][key]) for p in done])
                    p_val = (wilcoxon(d).pvalue if len(d) >= 6 and np.any(np.abs(d) > 1e-9)
                             else float("nan"))
                    line += f"{x.mean():8.2f} {d.mean():+7.2f} {int((d > 0).sum()):>3}/{len(d):<3}"
                    line += f"{p_val:8.1e}" if np.isfinite(p_val) else f"{'-':>8}"
                line += "".join(f"{np.mean([float(by[(p, v)][k]) for p in done]):{w}}"
                                for k, w in (("target_ratio", "8.3f"), ("seconds", "7.1f"),
                                             ("peak_gb", "6.2f")))
                print(line)


def main() -> None:
    # defaults first, so any --variants or --run given on the command line wins
    sys.argv = [sys.argv[0], "--variants", *ABLATION,
                "--run", f"ablation_{datetime.now():%m%d-%H%M}", *sys.argv[1:]]
    args = L.parse_args()
    L.main()
    per_cohort(args.out_dir / "results.csv", args.run, args.variants)


if __name__ == "__main__":
    main()
