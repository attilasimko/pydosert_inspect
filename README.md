# PyDoseRT validation workspace

Recompute clinical treatment-plan doses with [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT)
and compare them to what the planning system delivered, on two cohorts:

- **Vienna** — prostate, NIfTI + `plan_*.json`, Elekta Agility (`elekta_10MV`)
- **GoldAtlas** — DICOM, the Umeå 10 MV plans (`varian_10MV`), 10 MV arcs only

Two files:

| file | what it does |
|---|---|
| `loader.py` | reads a patient of either cohort onto a common 2 mm grid and returns one `Scan`: density, clinical total dose, beams, masks, body, target |
| `multilattice_lab.py` | builds the engine variants, computes dose, scores gamma, draws the figures, appends every result to `lab/results.csv` |

## Install

```bash
pip install pydosert SimpleITK matplotlib numpy scipy torch pymedphys pydicom
```

A CUDA GPU is expected; it falls back to CPU, which is slow.

## Run

```bash
python multilattice_lab.py                          # default variants on GoldAtlas
python multilattice_lab.py --variants L3_kernel L3_kernel_wide
python multilattice_lab.py --vienna /path/to/vienna --goldatlas ""
python multilattice_lab.py --limit 2 --beam_maps    # per-control-point mosaics
python multilattice_lab.py --list                   # the available variants
```

Outputs, all under `--out_dir` (default `lab/`):

- `results.csv` — one row per patient × variant, **appended** so runs accumulate, each
  row stamped with the pydosert revision and whether its working tree was edited
- `<run>.png` — gamma box plots per variant: whole body, skin shell, deeper
- `gamma_maps/<cohort>_<patient>.png` — dose, gamma and failure-overlap-vs-baseline
- `beam_maps/<patient>_arc<N>.png` — every control point's dose alone, one column per
  variant (`--beam_maps`)

## Variants

`VARIANTS` in `multilattice_lab.py` maps a name to engine settings; add a line to test
a new one. `baseline_k25` (one central-axis ray for the whole field) and `L1_mu0` (the
multilattice reduced to a single tile) are forced into every run as controls — the
multilattice is supposed to beat the first, and should closely reproduce it through the
second.

`mu_eff` is the residual depth correction applied because a tile's ray is not the voxel
being scored: `None` is pydosert's own (each tile's depth dose at its own field size),
a float is a constant attenuation per cm of water, and `0.0` switches it off so only
the per-tile ray geometry is left.

## Good to know

- Vienna CT/masks and dose come on **different grids**; everything is resampled in
  physical space onto one 2 mm isotropic grid first.
- Both cohorts are **padded to a cylinder** about the isocentre, so nothing leaves the
  array when a slice is rotated to a gantry angle.
- `BeamMeterset` is **per-fraction** MU, so the engine dose is multiplied by the number
  of fractions to compare with the clinical dose.
- The couch is **kept** by default (`--ct_mask none`): the clinical dose was computed
  with the table in place, and stripping it puts a failing band along the posterior skin.
- Vienna's `External` is not the patient — it holds body, air gap and couch (44.7 L vs
  26.4 L for `Body`), so the evaluation region is `Body`, rebuilt from the CT where the
  RTSTRUCT has none (GoldAtlas).
