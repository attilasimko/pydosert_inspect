"""Compare PyDoseRT engine variants against the clinical dose, and log the results.

    python engine_lab.py                              # default variants, GoldAtlas
    python engine_lab.py --variants cc12 cc24 cc48
    python engine_lab.py --cohort vienna --variants baseline_k25
    python engine_lab.py --limit 2 --beam_maps        # per-control-point mosaics
    python engine_lab.py --list                       # what the variants are

The claim under test is that the collapsed-cone correction is MORE accurate than the
bare pencil beam, which deposits dose plane by plane and can never move energy along
the beam in response to a density change. The correction is a ratio of two cone
transports, the real patient over a homogeneous one, so in uniform water it is
identically 1 and the engine reproduces the baseline exactly. Any run containing a
collapsed-cone variant therefore also gets two controls: `baseline_k25`, and `cc_off`
-- the collapsed-cone engine with `apply_correction=False`, which must come out
bit-for-bit equal to it. If it does not, the plumbing is wrong and no variant's number
means anything. A run of baselines alone gets no controls added, and --no_controls
turns them off entirely.

The engines keep changing, so a number only means something together with the engine
that produced it. Every logged row carries the pydosert revision, whether its working
tree was edited, and the variant's engine settings as JSON, and rows are APPENDED to
<out_dir>/results.csv, so runs accumulate instead of overwriting each other.

Scored inside the patient only: gamma over the whole body, over a skin shell
(--shell_mm) and over everything deeper -- the heterogeneity correction should show up
in the shell (build-up, oblique entry) and around the bones and bowel gas.

Data loading lives in loader.py; this file is engines, scoring and figures.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
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

# name -> engine settings. "kind" picks the engine -- "baseline" DoseEngine, "pencil"
# PencilDepthEngine, "cc" CollapsedConeEngine -- and "kernel" the pencil-beam kernel
# size; everything else is passed straight to the engine, so anything left out keeps
# the engine's own default. The collapsed-cone knobs are n_cones (transport
# directions), correction_clamp (bounds on the ratio), body_threshold (the density
# above which a voxel counts as patient in the homogeneous reference) and
# apply_correction (off = the baseline, exactly).
VARIANTS: dict[str, dict] = {
    # the published baseline: the engine's default depth cutoff is now 0, so the
    # 0.5 mm it was written with is pinned here to keep old runs comparable
    "baseline_k25": dict(kind="baseline", kernel=25, depth_threshold_mm=0.5),
    "baseline_k5": dict(kind="baseline", kernel=5, depth_threshold_mm=0.5),
    # the upgraded pencil beam, PencilDepthEngine: FFT convolution, so a 51-px (+-50 mm)
    # kernel costs no more than 25 px; per-pencil radiological depth; no depth cutoff
    # (its default); the machine's electron contamination. fft_k51 is only the larger
    # kernel, on DoseEngine -- ~3x faster.
    "pencil_k51": dict(kind="pencil", kernel=51, electron_contamination=True),
    "fft_k51": dict(kind="baseline", kernel=51, conv_backend="fft", depth_threshold_mm=0.5),
    # its ablation (ablation.py): fft_k51 with exactly one of pencil_k51's other changes
    "abl_pencil": dict(kind="pencil", kernel=51, depth_threshold_mm=0.5),
    "abl_no_cutoff": dict(kind="baseline", kernel=51, conv_backend="fft", depth_threshold_mm=0.0),
    "abl_contam": dict(kind="baseline", kernel=51, conv_backend="fft", depth_threshold_mm=0.5,
                       electron_contamination=True),
    # both of those together: pencil_k51 without the per-pencil depth
    "fft_k51_nc_contam": dict(kind="baseline", kernel=51, conv_backend="fft",
                              depth_threshold_mm=0.0, electron_contamination=True),
    "cc_off": dict(kind="cc", kernel=25, apply_correction=False),
    "cc12": dict(kind="cc", kernel=25, n_cones=12),
    "cc24": dict(kind="cc", kernel=25, n_cones=24),
    "cc48": dict(kind="cc", kernel=25, n_cones=48),
    "cc24_wide": dict(kind="cc", kernel=25, n_cones=24, correction_clamp=(0.2, 5.0)),
    "cc24_body05": dict(kind="cc", kernel=25, n_cones=24, body_threshold=0.5),
    "cc24_s025": dict(kind="cc", kernel=25, n_cones=24, correction_strength=0.25),
    "cc24_s050": dict(kind="cc", kernel=25, n_cones=24, correction_strength=0.50),
}
DEFAULT = ["cc24"]
CONTROLS = ["baseline_k25", "cc_off"]

GOLDATLAS_ROOT = Path("/home/bolo/Documents/PyDoseRT/test_data/GoldAtlasPlans/10X")
# 2021 harmonized export, Elekta Versa HD ("Versa_E" in the plan JSON), all 10 MV,
# one "Case 1" per patient, prescriptions of 20 and 7 fractions
VIENNA_ROOT = Path("/media/bolo/f4616a95-e470-4c0f-a21e-a75a8d283b9e/RAW/nifti_harmonized")
# knobs forwarded to the engine; everything else in a variant is for this script
ENGINE_KEYS = ("n_cones", "point_kernel", "correction_clamp", "body_threshold",
               "apply_correction", "correction_strength", "conv_backend",
               "depth_threshold_mm", "electron_contamination", "depth_nodes_mm")

FIELDS = ["run", "pydosert_rev", "dirty", "timestamp", "cohort", "patient", "variant",
          "kind", "kernel", "settings", "gamma", "gamma_shell", "gamma_deep", "mae_Gy",
          "target_ratio", "max_err_pct", "seconds", "peak_gb"]


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

def build_engine(v: dict, beam_sequence, shape, machine: str, device, args,
                 beam_chunk: int, cone_chunk: int):
    """One engine, built and calibrated, for the variant `v`.

    Calibrated after construction rather than with auto_calibrate=True, so every
    variant takes an identical path and calibration is not a confound. calibrate()
    ends by clearing layers_initialized, so the real beam template is rebuilt on the
    next compute_dose. The calibration phantom is uniform water, where the
    collapsed-cone ratio is identically 1, so all variants calibrate to the same
    absolute output.

    Every engine gets the same beam_chunk_size. The collapsed-cone engine needs it
    at least as much as the baseline: it asks the base class for its intermediates,
    which suppresses their early release, and it then rotates the chunk's fluence
    into the patient frame on top of that. Its correction is computed once per
    chunk from the TERMA summed over the chunk's beams, and since the transport is
    linear in TERMA that equals the dose-weighted mean of the per-beam corrections
    -- so the chunk size is a memory knob here, not a physics one.
    """
    common = dict(machine_config=MachineConfig(preset=machine),
                  kernel_size=v["kernel"],
                  dose_grid_spacing=(SPACING_MM,) * 3,
                  dose_grid_shape=tuple(shape),
                  beam_template=beam_sequence,
                  auto_calibrate=False,
                  dtype=args.dtype,
                  device=device)
    if v["kind"] in ("baseline", "pencil"):
        cls = PDRT.DoseEngine if v["kind"] == "baseline" else PDRT.PencilDepthEngine
        engine = cls(**common, beam_chunk_size=beam_chunk,
                     **{k: v[k] for k in ENGINE_KEYS if k in v})
    else:
        # imported here, not at module scope, so the baseline still runs against a
        # pydosert without the collapsed-cone engine
        from pydosert.engine.collapsed_cone import CollapsedConeEngine

        engine = CollapsedConeEngine(**common, beam_chunk_size=beam_chunk,
                                     cone_chunk=cone_chunk,
                                     **{k: v[k] for k in ENGINE_KEYS if k in v})
    engine.calibrate(verbose=False)
    return engine


def compute_dose(v: dict, scan: Scan, device, args) -> np.ndarray:
    """Total dose in Gy, summed over every beam of the plan.

    On CUDA OOM a memory knob is backed off and the whole calculation retried:
    beam_chunk_size bounds the [B*G,D,H,W] tensors that dominate both engines;
    cone_chunk bounds the cone directions transported at once, and is tried first
    for the collapsed cone because it is the cheaper of the two to shrink. A sweep
    meets a range of grid sizes and one fixed setting will not fit all of them.
    """
    beam_chunk, cone_chunk = args.beam_chunk_size, args.cone_chunk
    while True:
        try:
            total = None
            for bs in scan.beams:
                engine = build_engine(v, bs, scan.density.shape, scan.case.machine,
                                      device, args, beam_chunk, cone_chunk)
                with torch.no_grad():
                    d = engine.compute_dose(bs, density_image=scan.density)[0].float()
                total = d if total is None else total + d
                del engine, d
                free()
            return (total * scan.fractions).cpu().numpy()
        except torch.OutOfMemoryError:
            total = None
            free()
            if v["kind"] == "cc" and cone_chunk > 1:
                cone_chunk = max(1, cone_chunk // 2)
            elif beam_chunk > 1:
                beam_chunk = max(1, beam_chunk // 2)
            else:
                raise
            print(f"      OOM -> retry with beam_chunk={beam_chunk} "
                  f"cone_chunk={cone_chunk}", flush=True)


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

    Summed over an arc, a heterogeneity effect at one entry angle is smeared across
    hundreds of gantry angles; a single beam shows exactly where each variant deposits
    dose relative to the skin (white line) and the bones. The axial plane through the
    isocentre holds every beam's central plane, since the gantry rotates about that
    axis. There is no clinical dose per control point, so each row is scaled to its own
    maximum across the variants, which keeps the columns of a row comparable.

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
                              scan.density.shape, scan.case.machine, device, args,
                              1, args.cone_chunk)
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
    body = scan.body
    scan.clinical = clinical = np.where(body, scan.clinical, 0.0)
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
            pred = compute_dose(v, scan, device, args)
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
               "kind": v["kind"], "kernel": v["kernel"],
               "settings": json.dumps({k: v[k] for k in ENGINE_KEYS if k in v}),
               "gamma": pass_rate(g, body), "gamma_shell": pass_rate(g, shell),
               "gamma_deep": pass_rate(g, deep),
               "mae_Gy": float(np.abs(pred[hot] - clinical[hot]).mean()),
               "target_ratio": (float(pred[scan.target].mean() / clinical[scan.target].mean())
                                if scan.target is not None and scan.target.any() else float("nan")),
               "max_err_pct": 100.0 * (float(pred.max()) / float(clinical.max()) - 1.0),
               "seconds": round(seconds, 1), "peak_gb": round(peak_gb, 2)}
        writer.writerow({k: row[k] for k in FIELDS})
        rows.append(row)
        print(f"  {case.name:8s} {name:14s} gamma {row['gamma']:5.1f}%  shell {row['gamma_shell']:5.1f}%"
              f"  deep {row['gamma_deep']:5.1f}%  MAE {row['mae_Gy']:.3f} Gy"
              f"  {row['seconds']:6.1f}s", flush=True)
        del pred, g
        free()

    check_controls(slabs, case.name)
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


def check_controls(slabs: dict, patient: str) -> None:
    """cc_off must reproduce the baseline: with the correction off the collapsed-cone
    engine is the pencil beam. A difference here is plumbing, not physics."""
    a, b = CONTROLS
    if a in slabs and b in slabs:
        d = np.abs(slabs[a] - slabs[b]).max()
        scale = float(np.abs(slabs[a]).max()) or 1.0
        if d > 1e-4 * scale:
            print(f"  WARNING {patient}: {b} differs from {a} by {100 * d / scale:.3f}% "
                  f"of max dose; the two should be identical", flush=True)


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
    print(f"{'variant':<14}{'gamma':>8}{'shell':>8}{'deep':>8}{'MAE Gy':>9}{'max err':>9}"
          f"{'target':>8}{'sec':>8}{'GB':>6}")
    for v in variants:
        print(f"{v:<14}{mean(v, 'gamma'):8.2f}{mean(v, 'gamma_shell'):8.2f}"
              f"{mean(v, 'gamma_deep'):8.2f}{mean(v, 'mae_Gy'):9.3f}"
              f"{mean(v, 'max_err_pct'):8.1f}%{mean(v, 'target_ratio'):8.3f}"
              f"{mean(v, 'seconds'):8.1f}{mean(v, 'peak_gb'):6.2f}")

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
        print(f"  {v:<14} " + "  ".join(cells))

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


def prepare_log(path: Path, fields: list[str] | None = None) -> bool:
    """True if `path` needs a header written.

    Rows are appended so runs accumulate, which only works while the columns stay the
    same. When the engine knobs change the columns change with them, so a log written
    by an older version is moved aside rather than appended to with a different
    meaning per column.
    """
    if not path.exists() or path.stat().st_size == 0:
        return True
    with open(path, newline="") as f:
        header = next(csv.reader(f), [])
    if header == (fields or FIELDS):
        return False
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(path.stat().st_mtime))
    kept = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    path.rename(kept)
    print(f"{path.name} was written with different columns; kept as {kept.name}")
    return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default="goldatlas", choices=["goldatlas", "vienna", "both"],
                    help="which cohort to run; the roots below are the defaults for each")
    ap.add_argument("--goldatlas", type=Path, default=GOLDATLAS_ROOT)
    ap.add_argument("--vienna", type=Path, default=VIENNA_ROOT)
    ap.add_argument("--out_dir", type=Path, default=Path("lab"))
    ap.add_argument("--variants", nargs="*", default=DEFAULT,
                    help=f"any of {list(VARIANTS)}; {CONTROLS} are added alongside any "
                         f"collapsed-cone variant, as reference and plumbing check")
    ap.add_argument("--no_controls", action="store_true",
                    help="run exactly the variants asked for, controls included or not")
    ap.add_argument("--run", default=None, help="label for this run (default: rev + time)")
    ap.add_argument("--list", action="store_true", help="print the variants and exit")
    ap.add_argument("--limit", type=int, default=None, help="first N patients per cohort")
    ap.add_argument("--ct_mask", default="none", choices=["Body", "External", "none"],
                    help="ROI outside which the CT is set to air; 'none' (default) keeps "
                         "the couch, which the clinical dose was computed with")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    type=lambda s: getattr(torch, s),
                    help="engine precision; the collapsed-cone recursion is exercised "
                         "in float32 by pydosert's own tests")
    ap.add_argument("--shell_mm", type=float, default=25.0)
    ap.add_argument("--beam_chunk_size", type=int, default=4,
                    help="beams per chunk for both engines, halved on OOM")
    ap.add_argument("--cone_chunk", type=int, default=4,
                    help="cone directions transported at once; halved on OOM")
    ap.add_argument("--dose_pct", type=float, default=1.0)
    ap.add_argument("--dist_mm", type=float, default=1.0)
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
            print(f"  {name:<14} {settings}")
        return
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    # The controls are there to interpret a collapsed-cone number: baseline_k25 is what
    # it has to beat, cc_off is the plumbing check. A run of baselines only needs
    # neither, so asking for one variant then runs exactly that one variant.
    if not args.no_controls and any(VARIANTS[v]["kind"] == "cc" for v in args.variants):
        for c in reversed(CONTROLS):
            if c not in args.variants:
                args.variants.insert(0, c)

    rev, dirty = pydosert_revision()
    run = args.run or f"{rev}{'-dirty' if dirty else ''}_{datetime.now():%m%d-%H%M}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    roots = {"vienna": args.vienna if args.cohort in ("vienna", "both") else None,
             "goldatlas": args.goldatlas if args.cohort in ("goldatlas", "both") else None}
    cases = find_cases(roots["vienna"], roots["goldatlas"])
    if args.limit:
        cases = [c for cohort in ("vienna", "goldatlas")
                 for c in [x for x in cases if x.cohort == cohort][:args.limit]]
    if not cases:
        raise SystemExit("no patients under "
                         + " / ".join(str(r) for r in roots.values() if r is not None))
    meta = {"run": run, "pydosert_rev": rev, "dirty": int(dirty),
            "timestamp": datetime.now().isoformat(timespec="seconds")}
    print(f"run {run} | pydosert {rev}{' (working tree edited)' if dirty else ''} | "
          f"{len(cases)} patients | {device} | CT masked to {args.ct_mask}")
    print(f"variants: {args.variants}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / "results.csv"
    fresh = prepare_log(log)
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
