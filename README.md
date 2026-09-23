# PyDoseRT validation workspace

Recompute clinical treatment-plan doses with [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT)
and compare them to what the planning system delivered, on two cohorts:

- **Vienna** — prostate, NIfTI + `plan_*.json`, Elekta Agility (`elekta_10MV`)
- **GoldAtlas** — DICOM, the Umeå 10 MV plans (`varian_10MV`), 10 MV arcs only

Two files:

| file | what it does |
|---|---|
| `loader.py` | reads a patient of either cohort onto a common 2 mm grid and returns one `Scan`: density, clinical total dose, beams, masks, body, target |
| `engine_lab.py` | builds the engine variants, computes dose, scores gamma, draws the figures, appends every result to `lab/results.csv` |

## Install

```bash
pip install pydosert SimpleITK matplotlib numpy scipy torch pymedphys pydicom
```

A CUDA GPU is expected; it falls back to CPU, which is slow.

## Run

```bash
python engine_lab.py                          # default variants on GoldAtlas
python engine_lab.py --variants cc12 cc24 cc48
python engine_lab.py --vienna /path/to/vienna --goldatlas ""
python engine_lab.py --limit 2 --beam_maps    # per-control-point mosaics
python engine_lab.py --list                   # the available variants
```

Outputs, all under `--out_dir` (default `lab/`):

- `results.csv` — one row per patient × variant, **appended** so runs accumulate, each
  row stamped with the pydosert revision and whether its working tree was edited
- `<run>.png` — gamma box plots per variant: whole body, skin shell, deeper
- `gamma_maps/<cohort>_<patient>.png` — dose, gamma and failure-overlap-vs-baseline
- `beam_maps/<patient>_arc<N>.png` — every control point's dose alone, one column per
  variant (`--beam_maps`)

## Variants

`VARIANTS` in `engine_lab.py` maps a name to engine settings; add a line to test a new
one. Two controls are forced into every run:

- `baseline_k25` — the pencil beam the collapsed cone is supposed to beat
- `cc_off` — `CollapsedConeEngine(apply_correction=False)`, which **must** reproduce
  the baseline exactly; the run warns if it doesn't, because that's plumbing, not physics

The correction is a ratio, real patient over homogeneous patient, so it is identically
1 in water and leaves the commissioned output alone. Its knobs: `n_cones` (transport
directions, cost is linear in it), `correction_clamp` (bounds on the ratio),
`body_threshold` (density above which a voxel counts as patient in the homogeneous
reference).

Both engines take the same `--beam_chunk_size`. The collapsed cone computes its
correction once per chunk from the chunk's summed TERMA; the transport is linear in
TERMA, so the chunk size is a memory knob there, not a physics one. `--cone_chunk` is
its second memory knob, backed off first on OOM.

`pencil_k51` is the upgraded pencil beam, `PencilDepthEngine`: FFT convolution with a 51-px (±50 mm) kernel,
a radiological depth per pencil instead of one per depth plane, no 0.5 mm depth cutoff,
and the machine preset's electron-contamination term. `fft_k51` is only the larger
kernel, about three times faster.

`ablation.py` runs `baseline_k25`, then `fft_k51` alone and with each of the other
changes on its own (`abl_*`), and the full `pencil_k51`. It adds per-cohort tables with
paired Wilcoxon tests against `baseline_k25` and against `fft_k51`:

```bash
python ablation.py --limit 3                    # smoke test, GoldAtlas
python ablation.py --cohort both                # everything
```

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
