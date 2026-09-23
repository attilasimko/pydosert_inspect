"""Recompute every Vienna patient's dose and write it beside the clinical one.

    python save_vienna_doses.py                       # whole cohort -> vienna_doses/
    python save_vienna_doses.py --limit 3
    python save_vienna_doses.py --variant pencil_k51 --out_dir /tmp/doses

The output is the clinical dose's own file: same grid, spacing, origin, direction and
pixel type as Dose_0, written as <patient>/<case>/Dose_pydosert_<uid>.nii.gz with the
clinical file's UID kept, so the pair opens on top of each other in any viewer and
subtracts without resampling.

Dose is computed on the padded 2 mm grid the engines need and resampled back, which is
the one lossy step; it is the same interpolation the clinical dose went through to be
compared in the first place. Total dose in Gy, so directly comparable to Dose_0.

Each patient is also scored against its clinical dose as it is written -- gamma over
the whole body, over a skin shell (--shell_mm) and deeper -- on the computation grid,
inside the body, exactly as engine_lab does it, so the numbers land next to a
results.csv row for the same variant. --no_gamma skips it.

The engine is `fft_k51_nc_contam` from engine_lab.VARIANTS: FFT convolution with the
51-px kernel, no radiological-depth cutoff, and the machine's electron contamination
-- pencil_k51 without the per-pencil depth. Any other variant works via --variant.
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi

import engine_lab as L
from loader import SPACING_MM, find_cases, load_case, vienna_dose_path

FIELDS = ["patient", "variant", "fractions", "gamma", "gamma_shell", "gamma_deep",
          "pred_max", "clinical_max", "target_ratio", "seconds", "path"]


def write_like_clinical(dose: np.ndarray, grid, clinical_path: str, out_path: Path) -> None:
    """Write `dose` (on `grid`) onto the clinical dose's grid, in its format."""
    img = sitk.GetImageFromArray(dose.astype(np.float32))
    img.SetSpacing(grid.GetSpacing())
    img.SetOrigin(grid.GetOrigin())
    img.SetDirection(grid.GetDirection())
    clinical = sitk.ReadImage(clinical_path)
    out = sitk.Resample(img, clinical, sitk.Transform(), sitk.sitkLinear, 0.0,
                        clinical.GetPixelID())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(out_path), useCompression=True)


def gamma_rates(dose: np.ndarray, scan, args) -> dict:
    """Gamma pass rates inside the patient: whole body, skin shell, deeper.

    Scored on the computation grid rather than the written file, so a rate here means
    the same thing as the one engine_lab logs for the same variant. One gamma volume
    serves all three regions.
    """
    body = scan.body
    g = L.gamma_volume(np.where(body, scan.clinical, 0.0), np.where(body, dose, 0.0), args)
    shell = body & (ndi.distance_transform_edt(body, sampling=(SPACING_MM,) * 3)
                    <= args.shell_mm)
    return {"gamma": L.pass_rate(g, body), "gamma_shell": L.pass_rate(g, shell),
            "gamma_deep": L.pass_rate(g, body & ~shell)}


def out_path_for(case, clinical_path: str, out_dir: Path) -> Path:
    """<out_dir>/<the clinical dose's path under the cohort root>, renamed.

    Keeping the patient/case folders means --out_dir can be pointed at the cohort
    itself to write the doses in place, next to the clinical ones.
    """
    rel = Path(clinical_path).relative_to(case.path.parent)
    name = rel.name.replace("Dose_0_", "Dose_pydosert_", 1)
    if name == rel.name:                      # an export without a Dose_0
        name = f"Dose_pydosert_{rel.name.split('_', 1)[1]}"
    return out_dir / rel.parent / name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vienna", type=Path, default=L.VIENNA_ROOT)
    ap.add_argument("--out_dir", type=Path, default=Path("vienna_doses"))
    ap.add_argument("--variant", default="fft_k51_nc_contam", choices=list(L.VARIANTS))
    ap.add_argument("--limit", type=int, default=None, help="first N patients")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute patients whose dose file is already there")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    type=lambda s: getattr(torch, s))
    ap.add_argument("--beam_chunk_size", type=int, default=4, help="halved on OOM")
    ap.add_argument("--cone_chunk", type=int, default=4)
    ap.add_argument("--no_gamma", action="store_true", help="just write the doses")
    ap.add_argument("--shell_mm", type=float, default=25.0)
    ap.add_argument("--dose_pct", type=float, default=1.0)
    ap.add_argument("--dist_mm", type=float, default=1.0)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--subset", type=int, default=20000)
    args = ap.parse_args()

    v = L.VARIANTS[args.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = find_cases(args.vienna, None)[:args.limit]
    if not cases:
        raise SystemExit(f"no patients under {args.vienna}")
    rev, dirty = L.pydosert_revision()
    print(f"{len(cases)} patients | {device} | pydosert {rev}"
          f"{' (working tree edited)' if dirty else ''}")
    print(f"{args.variant}: {v}")

    if not args.no_gamma:
        print(f"gamma {args.dose_pct:g}%/{args.dist_mm:g}mm inside the body, "
              f"{args.shell_mm:g} mm shell")
    manifest = args.out_dir / "doses.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fresh = L.prepare_log(manifest, FIELDS)
    rows = []
    with open(manifest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for i, case in enumerate(cases, 1):
            clinical_path = vienna_dose_path(str(case.path))
            out = out_path_for(case, clinical_path, args.out_dir)
            if out.exists() and not args.overwrite:
                print(f"[{i}/{len(cases)}] {case.name}: already written")
                continue
            t0 = time.time()
            try:
                scan = load_case(case, device)
                dose = L.compute_dose(v, scan, device, args)
            except Exception:
                print(f"[{i}/{len(cases)}] {case.name}: FAILED")
                traceback.print_exc()
                L.free()
                continue
            write_like_clinical(dose, scan.grid, clinical_path, out)
            ratio = (float(dose[scan.target].mean() / scan.clinical[scan.target].mean())
                     if scan.target is not None and scan.target.any() else float("nan"))
            gammas = ({"gamma": float("nan"), "gamma_shell": float("nan"),
                       "gamma_deep": float("nan")} if args.no_gamma
                      else gamma_rates(dose, scan, args))
            row = {"patient": case.name, "variant": args.variant,
                   "fractions": scan.fractions, **gammas,
                   "pred_max": round(float(dose.max()), 3),
                   "clinical_max": round(float(scan.clinical.max()), 3),
                   "target_ratio": round(ratio, 4), "seconds": round(time.time() - t0, 1),
                   "path": str(out)}
            writer.writerow(row)
            f.flush()
            rows.append(row)
            print(f"[{i}/{len(cases)}] {case.name}: gamma {row['gamma']:5.1f}%  "
                  f"shell {row['gamma_shell']:5.1f}%  deep {row['gamma_deep']:5.1f}%  "
                  f"max {row['pred_max']:.2f} vs {row['clinical_max']:.2f} Gy  "
                  f"target {ratio:.3f}  {row['seconds']:.0f}s", flush=True)
            del scan, dose
            L.free()
    if rows and not args.no_gamma:
        fx = {r["fractions"] for r in rows}
        print(f"\nmean over {len(rows)} patients: "
              + "  ".join(f"{k} {np.mean([r[k] for r in rows]):.2f}%"
                          for k in ("gamma", "gamma_shell", "gamma_deep")))
        for n in sorted(fx):                 # the 7-fraction plans behave differently
            sub = [r for r in rows if r["fractions"] == n]
            print(f"  {n:>2}-fraction plans (n={len(sub)}): "
                  + "  ".join(f"{k} {np.mean([r[k] for r in sub]):.2f}%"
                              for k in ("gamma", "gamma_shell", "gamma_deep")))
    print(f"\nmanifest -> {manifest}")


if __name__ == "__main__":
    main()
