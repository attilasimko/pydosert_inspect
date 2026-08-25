"""Recompute a Vienna patient's plan dose with PyDoseRT and compare it to the ground truth.

    python main.py --patient_dir /path/to/vienna/0000IVBMJ

What it does, end to end:
    1. load the patient (CT + masks + ground-truth dose) and the treatment plan
    2. build the PyDoseRT engine and recompute the plan dose
    3. compare recomputed vs ground truth (PTV mean dose, DVHs, optional gamma)
    4. visualize the dose maps, the DVHs, and every intermediate stage of one beam
"""

import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from loader import load_patient, load_beam_sequence
from engine import build_engine, compute_dose, compute_intermediates

STRUCT_COLORS = {"PTV": "C3", "Rectum": "C1", "Bladder": "C0"}


def cumulative_dvh(dose, mask, dose_axis):
    """% of the structure receiving at least each dose in `dose_axis`."""
    values = dose[mask]
    return np.array([(values >= d).mean() * 100.0 for d in dose_axis])


def compute_gamma_map(gt, recomputed, z, spacing_mm=2.0, dose_pct=2, dist_mm=2):
    """A 2-D gamma map (2%/2mm) on axial slice z. Returns the gamma array (NaN below the 10% dose
    cutoff), or None if pymedphys is not installed."""
    try:
        import pymedphys
    except ImportError:
        print("   (install pymedphys to see the gamma map)")
        return None
    axes = tuple(np.arange(n) * spacing_mm for n in gt[z].shape)
    return pymedphys.gamma(axes, gt[z].astype(np.float64), axes, recomputed[z].astype(np.float64),
                           dose_percent_threshold=dose_pct, distance_mm_threshold=dist_mm,
                           lower_percent_dose_cutoff=10, max_gamma=2, quiet=True)


def plot_dose_maps(ct, gt, recomputed, z, gamma_map=None):
    """Axial maps at slice z: ground truth, PyDoseRT, their difference, and (if available) gamma."""
    vmax = float(np.percentile(gt, 99.9))
    diff = recomputed - gt
    dmax = max(abs(diff.min()), abs(diff.max())) or 1.0
    panels = [("ground truth (Gy)", gt[z], "jet", 0.0, vmax, 0.05 * vmax),
              ("PyDoseRT (Gy)", recomputed[z], "jet", 0.0, vmax, 0.05 * vmax),
              ("PyDoseRT - GT (Gy)", diff[z], "coolwarm", -dmax, dmax, 0.02 * dmax)]

    n = 4 if gamma_map is not None else 3
    fig, ax = plt.subplots(1, n, figsize=(4.6 * n, 4.6))
    for a, (title, img, cmap, lo, hi, thresh) in zip(ax, panels):
        a.imshow(ct[z], cmap="gray")
        im = a.imshow(np.ma.masked_where(np.abs(img) < thresh, img), cmap=cmap, vmin=lo, vmax=hi, alpha=0.65)
        a.set_title(title); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    if gamma_map is not None:
        valid = ~np.isnan(gamma_map)
        rate = 100.0 * np.mean(gamma_map[valid] <= 1) if valid.any() else 0.0
        a = ax[3]
        a.imshow(ct[z], cmap="gray")
        im = a.imshow(np.ma.masked_invalid(gamma_map), cmap="RdYlGn_r", vmin=0, vmax=2, alpha=0.75)
        a.set_title(f"gamma 2%/2mm  ({rate:.0f}% pass)\ngreen = pass (γ≤1), red = fail"); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.tight_layout()


def plot_dvh(gt, recomputed, masks):
    """DVHs for PTV / Rectum / Bladder: ground truth (solid) vs PyDoseRT (dashed)."""
    dmax = max(gt.max(), recomputed.max()) * 1.03      # headroom so the high-dose tail isn't clipped
    dose_axis = np.linspace(0.0, dmax, 250)
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for name, color in STRUCT_COLORS.items():
        if name in masks:
            m = masks[name]
            ax.plot(dose_axis, cumulative_dvh(gt, m, dose_axis), color=color, label=f"{name} (GT)")
            ax.plot(dose_axis, cumulative_dvh(recomputed, m, dose_axis), color=color, ls="--")
    ax.set_xlim(0, dmax); ax.set_ylim(0, 101)
    ax.set_xlabel("Dose (Gy)"); ax.set_ylabel("Volume (%)")
    ax.set_title("DVH  (solid = ground truth, dashed = PyDoseRT)"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()


def plot_intermediates(inter):
    """The stages the engine produces for one beam. Volumes are shown at their central slice; the
    1-D radiological-depth profile is shown as a line."""
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))

    def show(a, arr, title):
        arr = np.squeeze(arr)
        if arr.ndim <= 1:                                  # a profile, e.g. radiological depth vs depth
            a.plot(arr); a.set_xlabel("depth index"); a.set_ylabel("value")
            a.set_title(title, fontsize=10); a.grid(alpha=0.3)
            return
        if arr.ndim == 3:                                  # a volume -> central slice
            arr = arr[arr.shape[0] // 2]
        im = a.imshow(arr, cmap="viridis")
        a.set_title(title, fontsize=10); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)

    show(ax[0], inter["fluence_map"], "fluence map\n(MLC + jaw opening)")
    show(ax[1], inter["fluence_volume"], "fluence volume\n(central slice)")
    show(ax[2], inter["radiological_depth"], "radiological depth\n(depth profile)")
    show(ax[3], inter["beam_dose"], "beam dose\n(central slice)")
    fig.suptitle("PyDoseRT pipeline for one beam", y=1.02)
    fig.tight_layout()


def gamma_pass_rate(gt, recomputed, spacing_mm=2.0, dose_pct=2, dist_mm=2):
    """2%/2mm gamma pass rate via pymedphys (evaluated on a random subset for speed)."""
    import pymedphys
    axes = tuple(np.arange(n) * spacing_mm for n in gt.shape)
    g = pymedphys.gamma(axes, gt.astype(np.float64), axes, recomputed.astype(np.float64),
                        dose_percent_threshold=dose_pct, distance_mm_threshold=dist_mm,
                        lower_percent_dose_cutoff=10, max_gamma=2, random_subset=20000, quiet=True)
    valid = ~np.isnan(g)
    return 100.0 * np.mean(g[valid] <= 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patient_dir", required=True, help="a Vienna patient folder")
    ap.add_argument("--kernel_size", type=int, default=25, help="pencil-beam kernel width")
    ap.add_argument("--cp", type=int, default=None, help="control point to visualize (default: middle)")
    ap.add_argument("--gamma", action="store_true", help="also compute a 2%%/2mm gamma (needs pymedphys)")
    ap.add_argument("--save", action="store_true", help="save the figures as PNGs instead of showing them")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("1. loading patient and plan ...")
    patient, ref, masks = load_patient(args.patient_dir, device=device)
    beam_sequence, fractions, beams = load_beam_sequence(args.patient_dir, ref, device=device)
    ct = patient.density_image.cpu().numpy()
    gt = patient.dose.cpu().numpy()
    print(f"   grid {ct.shape}, {len(beams)} control points, {fractions} fractions")

    print("2. building engine and recomputing dose ...")
    engine = build_engine(patient, beam_sequence, kernel_size=args.kernel_size, device=device)
    recomputed = compute_dose(engine, beam_sequence, patient, fractions)

    print("3. comparing to ground truth ...")
    ptv = masks["PTV"]
    print(f"   PTV mean dose:  PyDoseRT {recomputed[ptv].mean():.1f} Gy   vs   ground truth {gt[ptv].mean():.1f} Gy")
    if args.gamma:
        print(f"   gamma 2%/2mm pass rate (3D): {gamma_pass_rate(gt, recomputed):.1f}%")

    print("4. visualizing ...")
    z = int(np.argwhere(masks["PTV"]).mean(axis=0)[0])  # a slice through the PTV centre
    gamma_map = compute_gamma_map(gt, recomputed, z)
    plot_dose_maps(ct, gt, recomputed, z, gamma_map)
    plot_dvh(gt, recomputed, masks)
    cp = args.cp if args.cp is not None else len(beams) // 2
    plot_intermediates(compute_intermediates(engine, beams[cp], patient))

    if args.save:
        for i, num in enumerate(plt.get_fignums()):
            plt.figure(num).savefig(f"demo_fig{i + 1}.png", dpi=110, bbox_inches="tight")
        print(f"   saved {len(plt.get_fignums())} figures (demo_fig*.png)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
