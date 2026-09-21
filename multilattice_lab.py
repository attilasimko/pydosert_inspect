"""Compare PyDoseRT engine variants against the clinical dose, and log the results.

    python multilattice_lab.py                          # default variants, GoldAtlas
    python multilattice_lab.py --variants L3_kernel L3_kernel_wide
    python multilattice_lab.py --vienna /path/to/vienna --goldatlas ""
    python multilattice_lab.py --limit 2 --beam_maps    # per-control-point mosaics
    python multilattice_lab.py --list                   # what the variants are

The claim under test is that the multilattice is MORE accurate than the baseline
pencil beam, which traces one central-axis ray for the whole field. Two controls are
therefore in every run: the baseline itself, and L=1 with no residual correction,
which is the same algorithm with its single ray moved to the centroid of the primary
fluence.

The engines keep changing, so a number only means something together with the engine
that produced it. Every logged row carries the pydosert revision and whether its
working tree was edited, and rows are APPENDED to <out_dir>/results.csv, so runs
accumulate instead of overwriting each other.

Scored inside the patient only: gamma over the whole body, over a skin shell
(--shell_mm, where the entry-angle correction acts) and over everything deeper. One
gamma volume per dose serves all three regions, and its random subset is seeded, so
paired differences between variants are not sampling noise.

Data loading lives in loader.py; this file is engines, scoring and figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import subprocess
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage as ndi

import pydosert as PDRT
from pydosert.data import MachineConfig

from loader import SPACING_MM, Scan, find_cases, load_case

# name -> engine settings. "kind" picks the engine, the rest are its knobs; anything
# left out takes the engine's own default. mu_eff is the residual depth correction:
# None is pydosert's own (each tile's depth dose at its own field size, no free
# parameter), a float is a constant attenuation per cm of water, and 0.0 switches it
# off so only the per-tile ray geometry is left.
VARIANTS: dict[str, dict] = {
    "baseline_k25": dict(kind="baseline", lattice=0, kernel=25),
    "baseline_k5": dict(kind="baseline", lattice=0, kernel=5),
    "L1_mu0": dict(kind="multilattice", lattice=1, kernel=25, mu_eff=0.0),
    "L1_kernel": dict(kind="multilattice", lattice=1, kernel=25, mu_eff=None),
    "L3_mu0": dict(kind="multilattice", lattice=3, kernel=25, mu_eff=0.0),
    "L3_kernel": dict(kind="multilattice", lattice=3, kernel=25, mu_eff=None),
    "L3_kernel_wide": dict(kind="multilattice", lattice=3, kernel=25, mu_eff=None,
                           cf_clamp=(0.05, 20.0)),
    "L3_mu005": dict(kind="multilattice", lattice=3, kernel=25, mu_eff=0.05),
    "L3_ss2": dict(kind="multilattice", lattice=3, kernel=25, mu_eff=None,
                   ray_supersample=2),
    "L5_mu0": dict(kind="multilattice", lattice=5, kernel=25, mu_eff=0.0),
    "L5_kernel": dict(kind="multilattice", lattice=5, kernel=25, mu_eff=None),
}
DEFAULT = ["L3_mu0", "L3_kernel", "L5_kernel"]
CONTROLS = ["baseline_k25", "L1_mu0"]

FIELDS = ["run", "pydosert_rev", "dirty", "timestamp", "cohort", "patient", "variant",
          "kind", "lattice", "kernel", "mu_eff", "cf_clamp", "ray_supersample",
          "gamma", "gamma_shell", "gamma_deep", "mae_Gy", "target_ratio", "max_err_pct",
          "seconds", "peak_gb"]


def free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def pydosert_revision() -> tuple[str, bool]:
    """Which engine produced these numbers: revision, and whether it was edited."""
    root = Path(PDRT.__file__).resolve().parents[2]

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(root), *a],
                              capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "--short", "HEAD") or "unknown", bool(git("status", "--porcelain"))


# ---------------------------------------------------------------------- the engines

def build_engine(v: dict, beam_sequence, shape, machine: str, device, beam_chunk: int,
                 tile_chunk: int):
    """One engine, built and calibrated, for the variant `v`.

    Calibrated after construction rather than with auto_calibrate=True, so both
    engines take an identical path and calibration is not a confound between
    variants. calibrate() ends by clearing layers_initialized, so the real beam
    template is rebuilt on the next compute_dose.
    """
    common = dict(machine_config=MachineConfig(preset=machine),
                  kernel_size=v["kernel"],
                  dose_grid_spacing=(SPACING_MM,) * 3,
                  dose_grid_shape=tuple(shape),
                  beam_template=beam_sequence,
                  auto_calibrate=False,
                  dtype=torch.float16,
                  device=device,
                  beam_chunk_size=beam_chunk)
    if v["kind"] == "baseline":
        engine = PDRT.DoseEngine(**common)
    else:
        # imported here, not at module scope, so the baseline still runs against a
        # pydosert without the multilattice engine
        from pydosert.engine.multilattice_engine import MultilatticeEngine

        extra = {k: v[k] for k in ("mu_eff", "cf_clamp", "ray_supersample") if k in v}
        engine = MultilatticeEngine(**common, lattice_size=max(v["lattice"], 1),
                                    tile_chunk=tile_chunk, **extra)
    engine.calibrate(verbose=False)
    return engine


def compute_dose(v: dict, scan: Scan, device, beam_chunk: int, tile_chunk: int) -> np.ndarray:
    """Total dose in Gy, summed over every beam of the plan.

    On CUDA OOM the two documented memory knobs are backed off and the whole
    calculation retried: beam_chunk_size first (it bounds the [B*G,D,H,W] tensors that
    dominate), then tile_chunk (tiles in flight, multilattice only). A sweep meets a
    range of grid sizes and one fixed pair of knobs will not fit all of them.
    """
    while True:
        try:
            total = None
            for bs in scan.beams:
                engine = build_engine(v, bs, scan.density.shape, scan.case.machine,
                                      device, beam_chunk, tile_chunk)
                with torch.no_grad():
                    d = engine.compute_dose(bs, density_image=scan.density)[0].float()
                total = d if total is None else total + d
                del engine, d
                free()
            return (total * scan.fractions).cpu().numpy()
        except torch.OutOfMemoryError:
            total = None
            free()
            if beam_chunk > 1:
                beam_chunk = max(1, beam_chunk // 2)
            elif v["kind"] != "baseline" and tile_chunk > 1:
                tile_chunk = max(1, tile_chunk // 2)
            else:
                raise
            print(f"      OOM -> retry with beam_chunk={beam_chunk} "
                  f"tile_chunk={tile_chunk}", flush=True)


# ---------------------------------------------------------------------- the scoring

def gamma_volume(gt: np.ndarray, pred: np.ndarray, args) -> np.ndarray:
    """3D gamma on a seeded random subset; unevaluated voxels are NaN.

    One call serves every region -- the subset is spread over the whole volume, so
    each region keeps a proportional share. pymedphys draws the subset with the global
    np.random.shuffle, and the seed makes every variant of a patient score the same
    voxels, so paired differences carry no sampling noise.
    """
    import pymedphys
    warnings.filterwarnings("ignore", message=".*quiet.*", category=DeprecationWarning)
    axes = tuple(np.arange(n) * SPACING_MM for n in gt.shape)
    state = np.random.get_state()
    np.random.seed(0)
    try:
        return pymedphys.gamma(axes, gt.astype(np.float64), axes, pred.astype(np.float64),
                               dose_percent_threshold=args.dose_pct,
                               distance_mm_threshold=args.dist_mm,
                               lower_percent_dose_cutoff=args.cutoff, max_gamma=2,
                               random_subset=args.subset, quiet=True)
    finally:
        np.random.set_state(state)


def pass_rate(g: np.ndarray, region: np.ndarray | None = None) -> float:
    valid = ~np.isnan(g)
    if region is not None:
        valid &= region
    return 100.0 * float(np.mean(g[valid] <= 1)) if valid.any() else float("nan")


def slice_gamma(clin_slab: np.ndarray, pred_slab: np.ndarray, k: int, z0: int, args,
                norm: float) -> np.ndarray:
    """3D gamma evaluated on one axial slice (index k within the slab), for the figures.

    The reference is the slice alone and the evaluation the whole slab around it, so
    the distance-to-agreement search still reaches the neighbouring slices; the dose
    criterion and cutoff are taken relative to the 3D maximum. That matches the
    whole-volume pass rates in results.csv rather than a stricter 2D gamma.
    """
    import pymedphys
    warnings.filterwarnings("ignore", message=".*quiet.*", category=DeprecationWarning)
    h, w = clin_slab.shape[1:]
    yx = (np.arange(h) * SPACING_MM, np.arange(w) * SPACING_MM)
    ref_axes = (np.array([(z0 + k) * SPACING_MM], dtype=np.float64),) + yx
    ev_axes = (np.arange(z0, z0 + pred_slab.shape[0]) * SPACING_MM,) + yx
    g = pymedphys.gamma(ref_axes, clin_slab[k:k + 1].astype(np.float64),
                        ev_axes, pred_slab.astype(np.float64),
                        dose_percent_threshold=args.dose_pct,
                        distance_mm_threshold=args.dist_mm,
                        lower_percent_dose_cutoff=args.cutoff, max_gamma=2,
                        global_normalisation=norm, quiet=True)
    return np.asarray(g)[0]


def pair_stats(ga: np.ndarray, gb: np.ndarray) -> dict:
    """Do two engines fail in the same voxels? Dice of the failing sets, r of gamma."""
    both = ~np.isnan(ga) & ~np.isnan(gb)
    fa, fb = both & (ga > 1), both & (gb > 1)
    n = int(fa.sum() + fb.sum())
    return {"dice": 2.0 * (fa & fb).sum() / n if n else float("nan"),
            "gamma_corr": (float(np.corrcoef(ga[both], gb[both])[0, 1])
                           if both.sum() > 10 else float("nan"))}


# ---------------------------------------------------------------------- the pictures

def gamma_figure(scan: Scan, z: int, doses: dict, gmaps: dict, crop: tuple,
                 out_path: Path, args) -> None:
    """Dose, gamma and failure overlap with the baseline, one column per variant."""
    y0, y1, x0, x1 = crop
    cut = lambda a: a[y0:y1, x0:x1]
    names = list(gmaps)
    base = CONTROLS[0] if CONTROLS[0] in gmaps else names[0]
    den = scan.density[z].float().cpu().numpy()
    clin = scan.clinical[z]
    vmax = float(scan.clinical.max()) or 1.0
    fig, ax = plt.subplots(3, len(names) + 1, figsize=(4.3 * (len(names) + 1), 12.5))

    ax[0, 0].imshow(cut(den), cmap="gray", vmin=0, vmax=1.5)
    ax[0, 0].imshow(np.ma.masked_where(cut(clin) < 0.05 * vmax, cut(clin)), cmap="jet",
                    vmin=0, vmax=vmax, alpha=0.6)
    ax[0, 0].set_title("clinical", fontsize=10)
    ax[1, 0].imshow(cut(den), cmap="gray", vmin=0, vmax=1.5)
    ax[1, 0].set_title("relative density", fontsize=10)
    ax[2, 0].axis("off")
    ax[2, 0].text(0.05, 0.6, f"failure overlap vs {base}\n\n"
                  f"blue    fails in {base} only\nred     fails in this variant only\n"
                  "purple  fails in both", fontsize=11, family="monospace", va="top",
                  transform=ax[2, 0].transAxes)

    bfail = gmaps[base] > 1
    for c, name in enumerate(names, start=1):
        a = ax[0, c]
        a.imshow(cut(den), cmap="gray", vmin=0, vmax=1.5)
        a.imshow(np.ma.masked_where(cut(doses[name]) < 0.05 * vmax, cut(doses[name])),
                 cmap="jet", vmin=0, vmax=vmax, alpha=0.6)
        a.set_title(name, fontsize=10)

        a = ax[1, c]
        a.imshow(cut(den), cmap="gray", vmin=0, vmax=1.5)
        im = a.imshow(np.ma.masked_invalid(cut(gmaps[name])), cmap="RdYlGn_r", vmin=0,
                      vmax=2, alpha=0.85)
        a.set_title(f"gamma {args.dose_pct:g}%/{args.dist_mm:g}mm, "
                    f"slice pass {pass_rate(gmaps[name]):.1f}%", fontsize=10)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

        a = ax[2, c]
        a.imshow(cut(den), cmap="gray", vmin=0, vmax=1.5)
        cfail = gmaps[name] > 1
        rgb = np.zeros(cut(den).shape + (4,))
        rgb[cut(bfail & ~cfail)] = (0.15, 0.4, 1.0, 0.9)
        rgb[cut(cfail & ~bfail)] = (1.0, 0.15, 0.15, 0.9)
        rgb[cut(cfail & bfail)] = (0.65, 0.15, 0.85, 0.9)
        a.imshow(rgb)
        if name == base:
            a.set_title(f"{name} fails: {int(bfail.sum())} voxels", fontsize=10)
        else:
            st = pair_stats(gmaps[base], gmaps[name])
            a.set_title(f"vs {base}: Dice {st['dice']:.2f}, r(gamma) {st['gamma_corr']:.2f}",
                        fontsize=10)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"{scan.case.cohort} {scan.case.name} — axial z={z}, inside the patient",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


MAX_IMAGE_PX = 60000        # matplotlib refuses images of 2**16 px or more per side


def _tile(den: np.ndarray, dose: np.ndarray, vmax: float, skin: np.ndarray) -> np.ndarray:
    """RGB tile: density in grey, dose in jet above 3% of vmax, skin line in white."""
    rgb = np.repeat(np.clip(den / 1.5, 0.0, 1.0)[..., None], 3, axis=2)
    frac = np.clip(dose / max(vmax, 1e-12), 0.0, 1.0)
    show = frac > 0.03
    rgb[show] = 0.35 * rgb[show] + 0.65 * plt.cm.jet(frac[show])[:, :3]
    rgb[skin] = 1.0
    return rgb


def beam_maps(scan: Scan, args, device, out_dir: Path) -> None:
    """Every control point's dose on its own, one row each, one column per variant.

    Summed over an arc, an entrance offset is smeared across hundreds of gantry
    angles; a single beam shows exactly where each variant starts depositing dose
    relative to the skin (white line). The axial plane through the isocentre holds
    every beam's central plane, since the gantry rotates about that axis. There is no
    clinical dose per control point, so each row is scaled to its own maximum across
    the variants, which keeps the columns of a row directly comparable.

    One engine per variant is built once and re-targeted per control point:
    compute_dose re-initialises the geometry for a new gantry angle without
    recalibrating, and gives the same dose as a freshly built engine.
    """
    cps = [(a, i) for a, bs in enumerate(scan.beams)
           for i in range(len(bs)) if float(bs.mus[i]) > 0.0][::max(1, args.beam_maps_stride)]
    if not cps:
        return
    z = int(round(float(scan.beams[0].iso_center[0]) / SPACING_MM))
    ys, xs = np.where(scan.body[z])
    y0, y1 = max(ys.min() - 6, 0), min(ys.max() + 7, scan.body.shape[1])
    x0, x1 = max(xs.min() - 6, 0), min(xs.max() + 7, scan.body.shape[2])
    den = scan.density[z].float().cpu().numpy()[y0:y1, x0:x1]
    skin = (scan.body[z] & ~ndi.binary_erosion(scan.body[z]))[y0:y1, x0:x1]
    h, w = den.shape

    doses = {}
    for name in args.variants:
        a0, i0 = cps[0]
        engine = build_engine(VARIANTS[name], scan.beams[a0].slice(i0, i0 + 1),
                              scan.density.shape, scan.case.machine, device,
                              args.beam_chunk_size, args.tile_chunk)
        out = np.empty((len(cps), h, w), dtype=np.float32)
        with torch.no_grad():
            for r, (a, i) in enumerate(cps):
                d = engine.compute_dose(scan.beams[a].slice(i, i + 1),
                                        density_image=scan.density, overwrite=True)[0]
                out[r] = d[z].float().cpu().numpy()[y0:y1, x0:x1]
        doses[name] = out
        del engine
        free()

    sep = 2
    rows_per_file = max(1, (MAX_IMAGE_PX - 200) // (h + sep))
    # one file per arc, split further only if an arc would exceed the image limit
    groups = []
    for a in sorted({a for a, _ in cps}):
        idx = [r for r, (aa, _) in enumerate(cps) if aa == a]
        groups += [(a, idx[k:k + rows_per_file]) for k in range(0, len(idx), rows_per_file)]

    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(doses)
    for g, (arc, idx) in enumerate(groups):
        H = len(idx) * (h + sep) + sep
        W = len(names) * (w + sep) + sep
        mosaic = np.zeros((H, W, 3), dtype=np.float32)
        labels = []
        for rr, r in enumerate(idx):
            a, i = cps[r]
            vmax = max(float(doses[n][r].max()) for n in names)
            for cc, n in enumerate(names):
                y, x = sep + rr * (h + sep), sep + cc * (w + sep)
                mosaic[y:y + h, x:x + w] = _tile(den, doses[n][r], vmax, skin)
            gdeg = float(np.rad2deg(float(scan.beams[a].gantry_angles[i]))) % 360.0
            labels.append(f"arc {a}  CP {i:3d}  {gdeg:5.1f}°  "
                          f"{float(scan.beams[a].mus[i]):5.2f} MU")
        dpi, left, top = 100, 240, 60
        fig = plt.figure(figsize=((W + left + 10) / dpi, (H + top + 10) / dpi), dpi=dpi)
        ax = fig.add_axes([left / (W + left + 10), 10 / (H + top + 10),
                           W / (W + left + 10), H / (H + top + 10)])
        ax.imshow(mosaic, interpolation="nearest", aspect="equal")
        ax.set_yticks([sep + rr * (h + sep) + h / 2 for rr in range(len(idx))], labels,
                      fontsize=7)
        ax.xaxis.tick_top()
        ax.set_xticks([sep + cc * (w + sep) + w / 2 for cc in range(len(names))], names,
                      fontsize=10)
        ax.tick_params(length=0)
        fig.suptitle(f"{scan.case.name}, arc {arc}: every control point alone, axial z={z} "
                     f"through the isocentre. Each row scaled to its own max; white = skin.",
                     fontsize=9, y=1 - 8 / (H + top + 10), va="top")
        part = "" if sum(1 for aa, _ in groups if aa == arc) == 1 else f"_part{g}"
        fig.savefig(out_dir / f"{scan.case.name}_arc{arc}{part}.png", dpi=dpi)
        plt.close(fig)


# ------------------------------------------------------------------------- the sweep

def run_patient(case, args, device, writer, meta: dict) -> list[dict]:
    scan = load_case(case, device, ct_mask=None if args.ct_mask == "none" else args.ct_mask)
    body, clinical = scan.body, np.where(scan.body, scan.clinical, 0.0)
    scan.clinical = clinical
    shell = body & (ndi.distance_transform_edt(body, sampling=(SPACING_MM,) * 3)
                    <= args.shell_mm)
    deep = body & ~shell
    hot = body & (clinical > 0.01 * clinical.max())
    z = (int(round(np.argwhere(scan.target).mean(axis=0)[0]))
         if scan.target is not None and scan.target.any()
         else int(np.argmax(clinical.reshape(clinical.shape[0], -1).max(1))))
    z0, z1 = max(z - args.slab, 0), min(z + args.slab + 1, clinical.shape[0])

    rows, slabs = [], {}
    for name in args.variants:
        v = VARIANTS[name]
        free()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            pred = compute_dose(v, scan, device, args.beam_chunk_size, args.tile_chunk)
        except Exception:
            print(f"  {case.name} {name}: FAILED")
            traceback.print_exc()
            free()
            continue
        pred = np.where(body, pred, 0.0)
        seconds, peak_gb = time.time() - t0, (torch.cuda.max_memory_allocated() / 2**30
                                              if device.type == "cuda" else float("nan"))
        slabs[name] = pred[z0:z1].copy()
        g = gamma_volume(clinical, pred, args)
        row = {**meta, "cohort": case.cohort, "patient": case.name, "variant": name,
               "kind": v["kind"], "lattice": v["lattice"], "kernel": v["kernel"],
               "mu_eff": v.get("mu_eff", "engine default"),
               "cf_clamp": v.get("cf_clamp", "engine default"),
               "ray_supersample": v.get("ray_supersample", 1),
               "gamma": pass_rate(g, body), "gamma_shell": pass_rate(g, shell),
               "gamma_deep": pass_rate(g, deep),
               "mae_Gy": float(np.abs(pred[hot] - clinical[hot]).mean()),
               "target_ratio": (float(pred[scan.target].mean() / clinical[scan.target].mean())
                                if scan.target is not None and scan.target.any() else float("nan")),
               "max_err_pct": 100.0 * (float(pred.max()) / float(clinical.max()) - 1.0),
               "seconds": round(seconds, 1), "peak_gb": round(peak_gb, 2)}
        writer.writerow({k: row[k] for k in FIELDS})
        rows.append(row)
        print(f"  {case.name:8s} {name:16s} gamma {row['gamma']:5.1f}%  shell {row['gamma_shell']:5.1f}%"
              f"  deep {row['gamma_deep']:5.1f}%  MAE {row['mae_Gy']:.3f} Gy"
              f"  {row['seconds']:5.1f}s", flush=True)
        del pred, g
        free()

    if slabs and not args.no_gamma_maps:
        norm = float(clinical.max())
        gmaps = {n: slice_gamma(clinical[z0:z1], sl, z - z0, z0, args, norm)
                 for n, sl in slabs.items()}
        ys, xs = np.where(body[z])
        crop = (max(ys.min() - 8, 0), min(ys.max() + 8, body.shape[1]),
                max(xs.min() - 8, 0), min(xs.max() + 8, body.shape[2]))
        maps = args.out_dir / "gamma_maps"
        maps.mkdir(parents=True, exist_ok=True)
        gamma_figure(scan, z, {n: sl[z - z0] for n, sl in slabs.items()}, gmaps, crop,
                     maps / f"{case.cohort}_{case.name}.png", args)
    del slabs
    if args.beam_maps:
        beam_maps(scan, args, device, args.out_dir / "beam_maps")
    del scan
    free()
    return rows


def summarize(rows: list[dict], args, run: str) -> None:
    variants = args.variants
    by = {(r["patient"], r["variant"]): r for r in rows}
    patients = sorted({r["patient"] for r in rows})
    done = [p for p in patients if all((p, v) in by for v in variants)]
    if not done:
        print("no patient completed every variant")
        return

    def mean(v: str, key: str) -> float:
        return float(np.mean([by[(p, v)][key] for p in done]))

    print(f"\n=== {run}: means over {len(done)} patients")
    print(f"{'variant':<16}{'gamma':>8}{'shell':>8}{'deep':>8}{'MAE Gy':>9}{'max err':>9}"
          f"{'target':>8}{'sec':>7}{'GB':>6}")
    for v in variants:
        print(f"{v:<16}{mean(v, 'gamma'):8.2f}{mean(v, 'gamma_shell'):8.2f}"
              f"{mean(v, 'gamma_deep'):8.2f}{mean(v, 'mae_Gy'):9.3f}"
              f"{mean(v, 'max_err_pct'):8.1f}%{mean(v, 'target_ratio'):8.3f}"
              f"{mean(v, 'seconds'):7.1f}{mean(v, 'peak_gb'):6.2f}")

    ref = CONTROLS[0] if CONTROLS[0] in variants else variants[0]
    print(f"\n=== paired against {ref}: mean gain in points (patients improved)")
    for v in variants:
        if v == ref:
            continue
        cells = []
        for key in ("gamma", "gamma_shell", "gamma_deep"):
            d = np.array([by[(p, v)][key] - by[(p, ref)][key] for p in done])
            cells.append(f"{key.replace('gamma_', ''):5s} {d.mean():+6.2f} "
                         f"({int((d > 0).sum())}/{len(d)})")
        print(f"  {v:<16} " + "  ".join(cells))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = {"gamma": "whole body", "gamma_shell": f"skin shell ({args.shell_mm:g} mm)",
              "gamma_deep": "deeper"}
    for a, key in zip(axes, titles):
        a.boxplot([[by[(p, v)][key] for p in done] for v in variants], labels=variants,
                  showmeans=True)
        a.axhline(mean(ref, key), color="C3", ls="--", lw=1, label=f"{ref} mean")
        a.set_title(titles[key])
        a.set_ylabel("% gamma pass")
        a.tick_params(axis="x", rotation=60, labelsize=8)
        a.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run} — {len(done)} patients, gamma {args.dose_pct:g}%/{args.dist_mm:g}mm",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = args.out_dir / f"{run}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goldatlas", type=Path,
                    default=Path("/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/10X"))
    ap.add_argument("--vienna", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("lab"))
    ap.add_argument("--variants", nargs="*", default=DEFAULT,
                    help=f"any of {list(VARIANTS)}; {CONTROLS} are always added")
    ap.add_argument("--run", default=None, help="label for this run (default: rev + time)")
    ap.add_argument("--list", action="store_true", help="print the variants and exit")
    ap.add_argument("--limit", type=int, default=None, help="first N patients per cohort")
    ap.add_argument("--ct_mask", default="none", choices=["Body", "External", "none"],
                    help="ROI outside which the CT is set to air; 'none' (default) keeps "
                         "the couch, which the clinical dose was computed with")
    ap.add_argument("--shell_mm", type=float, default=25.0)
    ap.add_argument("--beam_chunk_size", type=int, default=4, help="halved on OOM")
    ap.add_argument("--tile_chunk", type=int, default=4,
                    help="multilattice tiles in flight; halved on OOM once beam_chunk is 1")
    ap.add_argument("--dose_pct", type=float, default=2.0)
    ap.add_argument("--dist_mm", type=float, default=2.0)
    ap.add_argument("--cutoff", type=float, default=10.0)
    ap.add_argument("--subset", type=int, default=20000)
    ap.add_argument("--slab", type=int, default=3,
                    help="slices either side of the map slice searched by the 3D gamma")
    ap.add_argument("--no_gamma_maps", action="store_true")
    ap.add_argument("--beam_maps", action="store_true",
                    help="also draw every control point's dose on its own (slow)")
    ap.add_argument("--beam_maps_stride", type=int, default=10,
                    help="use every Nth control point in the beam maps")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        for name, settings in VARIANTS.items():
            print(f"  {name:<16} {settings}")
        return
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    for c in reversed(CONTROLS):                 # controls first, and always present
        if c not in args.variants:
            args.variants.insert(0, c)

    rev, dirty = pydosert_revision()
    run = args.run or f"{rev}{'-dirty' if dirty else ''}_{datetime.now():%m%d-%H%M}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = find_cases(args.vienna, args.goldatlas)
    if args.limit:
        cases = [c for cohort in ("vienna", "goldatlas")
                 for c in [x for x in cases if x.cohort == cohort][:args.limit]]
    if not cases:
        raise SystemExit(f"no patients under {args.vienna} / {args.goldatlas}")
    meta = {"run": run, "pydosert_rev": rev, "dirty": int(dirty),
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    print(f"run {run} | pydosert {rev}{' (working tree edited)' if dirty else ''} | "
          f"{len(cases)} patients | {device} | CT masked to {args.ct_mask}")
    print(f"variants: {args.variants}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "results.csv"
    fresh = not log.exists()
    rows = []
    with open(log, "a", newline="") as f:        # appended: runs accumulate
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if fresh:
            writer.writeheader()
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {case.cohort}/{case.name}")
            try:
                rows += run_patient(case, args, device, writer, meta)
            except Exception:
                print(f"  {case.name}: LOAD FAILED")
                traceback.print_exc()
            f.flush()

    summarize(rows, args, run)
    print(f"appended {len(rows)} rows to {log}")


if __name__ == "__main__":
    main()
