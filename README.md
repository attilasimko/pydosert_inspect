# PyDoseRT on the Vienna cohort — a minimal demo

A tiny, readable starting point for [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT): load one patient from the
Vienna prostate cohort, **recompute its treatment-plan dose**, and compare it to the clinical
ground-truth dose. It also visualizes every intermediate stage of the dose engine.

Three files, that's it:

| file | what it does |
|---|---|
| `loader.py` | reads a Vienna patient (CT, masks, dose) onto a common grid, and parses `plan.json` into a PyDoseRT `BeamSequence` |
| `engine.py` | builds the PyDoseRT `DoseEngine` (Elekta machine) and computes dose (full plan, or one beam with intermediates) |
| `main.py`   | runs the whole thing and makes the plots |

## Install

```bash
pip install pydosert SimpleITK matplotlib numpy torch
pip install pymedphys        # optional: enables the gamma map + the --gamma pass rate
```

A CUDA GPU is recommended (falls back to CPU, which is slow).

## Run

```bash
python main.py --patient_dir /path/to/vienna/0000IVBMJ
python main.py --patient_dir /path/to/vienna/0000IVBMJ --gamma      # + 2%/2mm gamma
python main.py --patient_dir /path/to/vienna/0000IVBMJ --cp 90      # visualize control point 90
```

A Vienna patient folder must contain `CT_*.nii.gz`, `mask_*_*.nii.gz`, `Dose_0_*.nii.gz`, and
`plan_*.json`.

## What you'll see

1. **Dose maps** — ground-truth dose, PyDoseRT-recomputed dose, their difference, and a **2%/2mm
   gamma map** (green = pass, red = fail), overlaid on the CT at a slice through the PTV.
2. **DVHs** — PTV / Rectum / Bladder, ground truth (solid) vs recomputed (dashed).
3. **Engine pipeline for one beam** — the four stages PyDoseRT goes through to turn leaf/jaw
   positions into dose:
   `fluence map` → `fluence volume` → `radiological depth` → `beam dose`.

## Good to know

- CT/masks and the dose come on **different grids**; `loader.py` resamples everything in physical
  space onto one common 2 mm isotropic grid first.
- The plan's `BeamMeterset` is **per-fraction** MU, so the engine dose is multiplied by the number
  of fractions to get the total dose comparable to `Dose_0`.
- The Vienna machine is an **Elekta Agility** (80 leaf pairs) → `MachineConfig(preset="elekta_10MV")`.
- `kernel_size` trades accuracy for speed (25 ≈ accurate; 5 ≈ fast/coarse).
