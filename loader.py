"""Load the Vienna and GoldAtlas patients onto a grid PyDoseRT can compute on.

    from loader import find_cases, load_case
    for case in find_cases(vienna_root, goldatlas_root):
        scan = load_case(case, device)

Everything downstream sees the same `Scan` whichever cohort it came from: relative
density, the clinical total dose in Gy, the beams, the masks, and the body and target
masks the evaluation needs.

The two cohorts arrive very differently.

Vienna is NIfTI plus a plan JSON. CT+masks and dose sit on DIFFERENT voxel grids, so
everything is resampled in physical (mm) space onto one common isotropic grid first.
GoldAtlas is DICOM, read by pydosert's own loader; its plan carries 6 MV setup fields
with no meterset next to the 10 MV treatment arcs, and only the arcs that deliver MU
are kept.

Both are then padded in the axial plane (pad_to_cylinder) so the isocentre sits at the
centre and the whole body stays inside the circle that every gantry angle keeps -- the
engines rotate each axial slice about the isocentre, and anything outside that circle
would leave the array at some angle and be treated as air.
"""

from __future__ import annotations

import dataclasses
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
from scipy import ndimage as ndi

import pydosert as PDRT
from pydosert.data import Patient, loaders
from pydosert.data.beam import Beam, BeamSequence

try:    # added to pydosert with the cylinder padding; some builds do not have it
    from pydosert.data.loaders import body_cylinder_radius_mm, pad_to_cylinder
except ImportError:
    def body_cylinder_radius_mm(ct_volume, resolution, iso_center, air_hu=-500.0):
        """Largest axial distance from the isocentre to a non-air voxel, in mm.

        Local stand-in for pydosert's, with the same semantics, so a build without it
        still pads identically. Voxel centres sit at (index + 0.5) * spacing.
        """
        arr = (ct_volume.detach().cpu().numpy() if isinstance(ct_volume, torch.Tensor)
               else np.asarray(ct_volume))
        body = (arr > air_hu).any(axis=0)
        if not body.any():
            return 0.0
        _, res_d, res_w = resolution
        d_idx, w_idx = np.nonzero(body)
        dd = (d_idx + 0.5) * res_d - iso_center[1]
        ww = (w_idx + 0.5) * res_w - iso_center[2]
        return float(np.sqrt(dd * dd + ww * ww).max())

    def pad_to_cylinder(volumes, resolution, iso_center, radius_mm=0.0, fill_value=0.0):
        """Pad D and W so the isocentre is centred and `radius_mm` fits. H is the
        rotation axis and is left alone. Local stand-in for pydosert's."""
        single = not isinstance(volumes, (list, tuple))
        vol_list = [volumes] if single else list(volumes)
        fills = (list(fill_value) if isinstance(fill_value, (list, tuple))
                 else [fill_value] * len(vol_list))
        if len(fills) != len(vol_list):
            raise ValueError(f"{len(fills)} fill values for {len(vol_list)} volumes")
        H, D, W = vol_list[0].shape[-3:]
        if any(tuple(v.shape[-3:]) != (H, D, W) for v in vol_list):
            raise ValueError(f"all volumes must share the (H, D, W) grid {(H, D, W)}")
        _, res_d, res_w = resolution
        _, iso_d, iso_w = iso_center

        def _centring(n, iso_voxels):
            diff = 2.0 * iso_voxels - n
            return (0, math.ceil(diff)) if diff >= 0 else (math.ceil(-diff), 0)

        d_before, d_after = _centring(D, iso_d / res_d)
        w_before, w_after = _centring(W, iso_w / res_w)
        if radius_mm:
            # +2 voxels: the radius is measured to voxel centres. Both sides grow
            # equally, or the isocentre would move off centre.
            need_d = math.ceil(2.0 * radius_mm / res_d) + 2 - (D + d_before + d_after)
            need_w = math.ceil(2.0 * radius_mm / res_w) + 2 - (W + w_before + w_after)
            if need_d > 0:
                d_before += math.ceil(need_d / 2.0)
                d_after += math.ceil(need_d / 2.0)
            if need_w > 0:
                w_before += math.ceil(need_w / 2.0)
                w_after += math.ceil(need_w / 2.0)

        def _pad(v, fv):
            if isinstance(v, torch.Tensor):
                return torch.nn.functional.pad(v, (w_before, w_after, d_before, d_after),
                                               mode="constant", value=float(fv))
            widths = [(0, 0)] * (v.ndim - 2) + [(d_before, d_after), (w_before, w_after)]
            return np.pad(v, widths, mode="constant", constant_values=fv)

        padded = [_pad(v, fv) for v, fv in zip(vol_list, fills)]
        new_iso = (iso_center[0], iso_d + d_before * res_d, iso_w + w_before * res_w)
        info = {"d_before": d_before, "w_before": w_before, "original_shape": (H, D, W)}
        if single:
            return padded[0], new_iso, info
        return (tuple(padded) if isinstance(volumes, tuple) else padded), new_iso, info


SPACING_MM = 2.0  # the common isotropic grid everything is resampled onto
VIENNA_MACHINE = "elekta_10MV"
GOLDATLAS_MACHINE = "varian_10MV"
# first match wins; the cohorts disagree on what the target is called
TARGET_NAMES = ["PTV", "PTVT", "CTVT", "CTV"]

# Vienna: nifti structure file name -> the name the mask is stored under
STRUCTURES = {
    "External": "External",   # the ROI RayStation reported dose in: body + air gap + couch
    "Body": "Body",           # the patient alone
    "PTV": "PTV",
    "CTV": "CTV",
    "Rectum": "Rectum",
    "Bladder": "Bladder",
    "Femur_Head_L": "FemoralHead_L",
    "Femur_Head_R": "FemoralHead_R",
}
# GoldAtlas: requested by name rather than with struct_names=None, which raises --
# load_structures builds a plain list on the None path and then iterates it as a dict.
# Naming them also fixes the keys: the target comes back as "PTVT" whether the RTSTRUCT
# calls it PTVT or PTVT_42.7. Only these two are used, and asking for the OARs as well
# would drag in "Rectum", which substring-matches z_opt_Rectum on three patients.
GOLDATLAS_STRUCTS = ["External", "PTVT"]


@dataclass
class Case:
    """One patient, resolved to whatever its cohort needs to load it."""
    cohort: str
    name: str
    path: Path
    machine: str


@dataclass
class Scan:
    """A loaded patient, the same shape of thing for both cohorts.

    density        relative density [D, H, W] on the device (air ~0, water ~1)
    clinical       clinical TOTAL dose in Gy, numpy, same grid
    beams          one BeamSequence per arc; dose is additive, so they are summed
    masks          structure masks on the device
    body           the patient alone, bool numpy -- the evaluation region
    target         the target mask, bool numpy, or None
    fractions      the plan's fractions (beam MU is per fraction)
    """
    case: Case
    density: torch.Tensor
    clinical: np.ndarray
    beams: list
    masks: dict
    body: np.ndarray
    target: np.ndarray | None
    target_name: str | None
    fractions: int


def find_cases(vienna_root: Path | None, goldatlas_root: Path | None) -> list[Case]:
    cases = []
    if vienna_root and Path(vienna_root).is_dir():
        for d in sorted(p for p in Path(vienna_root).iterdir() if p.is_dir()):
            cases.append(Case("vienna", d.name, d, VIENNA_MACHINE))
    if goldatlas_root and Path(goldatlas_root).is_dir():
        for d in sorted(p for p in Path(goldatlas_root).iterdir() if p.is_dir()):
            cases.append(Case("goldatlas", d.name, d, GOLDATLAS_MACHINE))
    return cases


def load_case(case: Case, device: torch.device, ct_mask: str | None = None) -> Scan:
    """Load one patient.

    ct_mask : ROI outside which the CT is set to air before dose calculation, or None
              (default) to keep everything, couch included. Stripping the couch put a
              failing band along the whole posterior skin on GoldAtlas: the clinical
              dose was computed with the table under the patient.
    """
    if case.cohort == "vienna":
        density, clinical, beams, masks, fractions = _load_vienna(case, device, ct_mask)
    else:
        density, clinical, beams, masks, fractions = _load_goldatlas(case, device, ct_mask)
    target_name, target_t = pick_mask(masks, TARGET_NAMES)
    return Scan(case=case, density=density, clinical=clinical, beams=beams, masks=masks,
                body=body_mask(density, masks),
                target=target_t.cpu().numpy().astype(bool) if target_t is not None else None,
                target_name=target_name, fractions=fractions)


def pick_mask(masks: dict, names: list[str]):
    """The first of `names` present, as (name, mask), or (None, None)."""
    for n in names:
        if n in masks:
            return n, masks[n]
    return None, None


def body_mask(density: torch.Tensor, masks: dict) -> np.ndarray:
    """The patient alone: the Body contour if the plan has one, else from the CT.

    GoldAtlas RTSTRUCTs have no Body, so it is rebuilt as the largest connected region
    of density > 0.5 with its cavities filled. Only used to bound the evaluation.
    """
    _, body_t = pick_mask(masks, ["Body"])
    if body_t is not None:
        return body_t.cpu().numpy().astype(bool)
    lab, _ = ndi.label(density.cpu().numpy() > 0.5)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    body = ndi.binary_fill_holes(lab == sizes.argmax())
    for z in range(body.shape[0]):             # cavities that only open along z
        body[z] = ndi.binary_fill_holes(body[z])
    return body


# --------------------------------------------------------------------------- Vienna

def _case_dir(patient_dir: str) -> str:
    """The folder holding a patient's files.

    Newer exports nest everything one level down in a case folder ("Case 1"); older
    ones put the files straight in the patient folder. Accept either.
    """
    if glob.glob(os.path.join(patient_dir, "CT_*.nii.gz")):
        return patient_dir
    cases = sorted(d for d in glob.glob(os.path.join(patient_dir, "*"))
                   if glob.glob(os.path.join(d, "CT_*.nii.gz")))
    return cases[0]


def _reference_grid(ct_img, spacing_mm: float):
    """A SimpleITK grid with the CT's physical extent but isotropic `spacing_mm` voxels."""
    size = [int(round(n * s / spacing_mm)) for n, s in zip(ct_img.GetSize(), ct_img.GetSpacing())]
    ref = sitk.Image(size, ct_img.GetPixelID())
    ref.SetSpacing((spacing_mm,) * 3)
    ref.SetOrigin(ct_img.GetOrigin())
    ref.SetDirection(ct_img.GetDirection())
    return ref


def _resample(img, ref, interpolator) -> np.ndarray:
    """Resample `img` onto the reference grid and return a numpy array [D, H, W]."""
    out = sitk.Resample(img, ref, sitk.Transform(), interpolator, 0.0, img.GetPixelID())
    return sitk.GetArrayFromImage(out)


def _plan_isocentre(case_dir: str):
    """The plan's DICOM isocentre (x, y, z mm). Every arc shares it: BeamSequence
    refuses to stack beams with different isocentres."""
    plan = json.load(open(sorted(glob.glob(os.path.join(case_dir, "plan_*.json")))[0]))
    return plan["BeamSequence"][0]["Isocenter"]


def _padded_reference(ref, pad_info: dict, shape):
    """The SimpleITK grid of a cylinder-padded volume.

    Same spacing and direction, origin moved back by the padding, so a physical point
    keeps its position: anything placed through it -- the plan isocentre, a dose
    resampled back onto the clinical grid -- stays right without further bookkeeping.
    """
    h, d, w = shape                                     # array (z, y, x) = sitk (x, y, z)
    out = sitk.Image([int(w), int(d), int(h)], ref.GetPixelID())
    out.SetSpacing(ref.GetSpacing())
    out.SetDirection(ref.GetDirection())
    out.SetOrigin(ref.TransformContinuousIndexToPhysicalPoint(
        (-float(pad_info["w_before"]), -float(pad_info["d_before"]), 0.0)))
    return out


def _isocentre_to_grid(iso_xyz_mm, ref, spacing_mm: float):
    """Map the plan's DICOM isocentre (x, y, z mm) to the engine's grid frame (z, y, x mm)."""
    ix, iy, iz = ref.TransformPhysicalPointToContinuousIndex([float(v) for v in iso_xyz_mm])
    return (iz * spacing_mm, iy * spacing_mm, ix * spacing_mm)


def load_patient(patient_dir: str, spacing_mm: float = SPACING_MM, device="cuda",
                 ct_mask: str | None = None):
    """Load a Vienna patient onto a common padded grid.

    Returns:
        patient : a pydosert Patient (holds the CT-derived density and the GT dose)
        ref     : the SimpleITK reference grid of the PADDED volume (needed to place
                  the plan isocentre, and to resample anything onto this grid)
        masks   : {name: bool array [D, H, W]} the structure masks, padded
    """
    case_dir = _case_dir(patient_dir)
    ct_img = sitk.ReadImage(sorted(glob.glob(os.path.join(case_dir, "CT_*.nii.gz")))[0])
    ref = _reference_grid(ct_img, spacing_mm)
    ct = _resample(ct_img, ref, sitk.sitkLinear).astype(np.float32)  # Hounsfield units

    masks = {}
    for fname, name in STRUCTURES.items():
        found = sorted(glob.glob(os.path.join(case_dir, f"mask_{fname}_*.nii.gz")))
        if found:  # nearest-neighbour keeps masks binary
            masks[name] = _resample(sitk.ReadImage(found[0]), ref, sitk.sitkNearestNeighbor) > 0.5

    dose_files = (sorted(glob.glob(os.path.join(case_dir, "Dose_0_*.nii.gz")))
                  or sorted(glob.glob(os.path.join(case_dir, "Dose_*.nii.gz"))))
    gt_dose = _resample(sitk.ReadImage(dose_files[0]), ref, sitk.sitkLinear).astype(np.float32)

    if ct_mask is not None:
        if ct_mask not in masks:
            raise KeyError(f"{patient_dir}: no {ct_mask} contour to mask the CT with; "
                           f"have {sorted(masks)}")
        # Air, not zero: HU 0 is water, so multiplying by the mask would fill the
        # couch and everything around the patient with water.
        ct = np.where(masks[ct_mask], ct, -1000.0).astype(np.float32)

    # Masking first means the padding only has to hold what is still there.
    resolution = (spacing_mm,) * 3
    iso = _isocentre_to_grid(_plan_isocentre(case_dir), ref, spacing_mm)
    radius = body_cylinder_radius_mm(ct, resolution, iso)
    names = list(masks)
    padded, new_iso, pad_info = pad_to_cylinder(
        [ct, gt_dose] + [masks[n] for n in names], resolution, iso, radius,
        fill_value=[-1000.0, 0.0] + [0] * len(names))
    ct, gt_dose = padded[0], padded[1]
    masks = {n: padded[2 + k] for k, n in enumerate(names)}
    ref = _padded_reference(ref, pad_info, ct.shape)
    # the padded grid must put the plan's isocentre exactly where the util moved it
    check = _isocentre_to_grid(_plan_isocentre(case_dir), ref, spacing_mm)
    if not np.allclose(check, new_iso, atol=1e-3):
        raise RuntimeError(f"padded grid puts the isocentre at {check}, "
                           f"pad_to_cylinder at {new_iso}")

    # pydosert's Patient converts the HU CT to physical density internally.
    patient = PDRT.Patient(ct_tensor=torch.from_numpy(ct), dose=torch.from_numpy(gt_dose),
                           resolution=resolution)
    for name, mask in masks.items():
        patient.add_mask(name, torch.from_numpy(np.ascontiguousarray(mask)))
    return patient.to(device).to(torch.float32), ref, masks


def load_beam_sequence(patient_dir: str, ref, spacing_mm: float = SPACING_MM, device="cuda"):
    """Parse plan_*.json into a pydosert BeamSequence (one Beam per control point).

    Returns:
        beam_sequence : the full BeamSequence (all arcs, all control points)
        fractions     : number of fractions. BeamMeterset is PER-FRACTION MU, so the
                        engine dose must be multiplied by `fractions` to get the total.
        beams         : the list of individual Beam objects (handy for per-beam demos)
    """
    plan_path = sorted(glob.glob(os.path.join(_case_dir(patient_dir), "plan_*.json")))[0]
    plan = json.load(open(plan_path))
    fractions = int(round(float(plan.get("Fractions", 1))))

    beams = []
    for arc in plan["BeamSequence"]:
        gantry = np.asarray(arc["GantryAngles"], dtype=np.float64)               # deg, per CP
        cum_mu = np.asarray(arc["CumulativeMetersetWeights"], dtype=np.float64)  # 0..1, per CP
        leaves = np.asarray(arc["MLCX"], dtype=np.float32)                       # [CP, N, 2] mm
        jaws = np.asarray(arc["ASYMY"], dtype=np.float32)                        # [CP, 2] mm
        ssd = np.asarray(arc.get("SourceToSurfaceDistances", [1000.0] * len(gantry)),
                         dtype=np.float64)
        meterset = float(arc["BeamMeterset"])                                    # per-fraction MU
        sad = float(arc["SourceAxisDistance"])
        collimator = math.radians(float(arc.get("BeamLimitingDeviceAngle", 0.0)))
        iso = _isocentre_to_grid(arc["Isocenter"], ref, spacing_mm)
        mu = np.diff(cum_mu, prepend=0.0) * meterset     # cumulative weight -> MU per CP

        for i in range(len(gantry)):
            beams.append(Beam(
                gantry_angle=math.radians(float(gantry[i])),
                collimator_angle=collimator,
                ssd=float(ssd[i]),
                mu=torch.tensor(float(mu[i])),
                leaf_positions=torch.from_numpy(leaves[i]),  # [N_leaves, 2]
                jaw_positions=torch.from_numpy(jaws[i]),     # [2]
                field_size=(400, 400),
                sid=sad,
                iso_center=iso,
            ))
    return BeamSequence.from_beams(beams).to(device).to(torch.float32), fractions, beams


def _load_vienna(case: Case, device, ct_mask):
    patient, ref, masks = load_patient(str(case.path), device=device, ct_mask=ct_mask)
    beam_sequence, fractions, _ = load_beam_sequence(str(case.path), ref, device=device)
    masks = {k: torch.from_numpy(v).to(device) for k, v in masks.items()}
    # Dose_0 is already the total dose
    return patient.density_image, patient.dose.cpu().numpy(), [beam_sequence], masks, fractions


# ------------------------------------------------------------------------ GoldAtlas

def _one(paths: list[Path], what: str, patient: str) -> Path:
    if not paths:
        raise FileNotFoundError(f"{patient}: no {what}")
    return paths[0]


def _load_goldatlas(case: Case, device, ct_mask):
    ct_dir = _one(sorted(case.path.glob("[[]CT[]]*")), "CT folder", case.name)
    plan = _one(sorted(case.path.glob("[[]RP[]]*/*.dcm")), "RTPLAN", case.name)
    dose = _one(sorted(case.path.glob("[[]RD[]]*/*.dcm")), "RTDOSE", case.name)
    struct = _one(sorted(case.path.glob("[[]RS[]]*/*.dcm")), "RTSTRUCT", case.name)
    # crop_volume=False: no 40 cm centre crop -- the cylinder padding below is what
    # keeps the body inside every rotation, and a crop can only take tissue away.
    patient, beam_sequences = loaders.load_dicom(
        ct_folder=ct_dir, dose_path=dose, plan_path=plan, struct_path=struct,
        struct_names=GOLDATLAS_STRUCTS, use_delivery=True, crop_volume=False,
        new_spacing=(SPACING_MM,) * 3, device=device)
    fractions = int(patient.number_of_fractions)

    # Keep only the treatment arcs. The 6 MV setup fields carry no meterset, so
    # filtering on delivered MU drops exactly those and leaves the 10 MV arcs.
    kept = [s.to(device).to(torch.float32) for s in beam_sequences if float(s.mus.sum()) > 0.0]
    if not kept:
        raise ValueError(f"{case.name}: no beam carries MU")
    check_energies(plan, case.name)

    # One padding serves all arcs only because they share an isocentre, so check that.
    isos = {tuple(round(float(v), 3) for v in s.iso_center) for s in kept}
    if len(isos) != 1:
        raise ValueError(f"{case.name}: arcs have different isocentres {isos}")
    iso = tuple(float(v) for v in kept[0].iso_center)
    # _ct_tensor is the HU volume (Patient has no public accessor); these CT series
    # read as float64, which would reach the convolution as Double against Half kernels
    ct = patient._ct_tensor.to(torch.float32)
    if ct_mask is not None:
        # GoldAtlas has no Body ROI; its External is the patient (it matches the CT's
        # body to within 0.3%), so it stands in for Body. Air, not zero: HU 0 is water.
        roi = "External" if ct_mask == "Body" else ct_mask
        if roi not in patient.structures:
            raise KeyError(f"{case.name}: no {roi} to mask the CT with; "
                           f"have {sorted(patient.structures)}")
        ct = torch.where(patient.structures[roi].to(ct.device), ct, torch.full_like(ct, -1000.0))

    names = list(patient.structures)
    radius = body_cylinder_radius_mm(ct, patient.resolution, iso)
    padded, new_iso, _ = pad_to_cylinder(
        [ct, patient.dose.to(torch.float32)] + [patient.structures[n] for n in names],
        patient.resolution, iso, radius, fill_value=[-1000.0, 0.0] + [0] * len(names))
    patient = Patient(ct_tensor=padded[0], dose=padded[1],
                      structures={n: padded[2 + k] for k, n in enumerate(names)},
                      resolution=patient.resolution,
                      number_of_fractions=fractions).to(device).to(torch.float32)
    # rebuilt, not assigned: BeamSequence is a frozen dataclass in current pydosert
    kept = [dataclasses.replace(s, iso_center=new_iso) for s in kept]

    masks = {k: v.to(device) for k, v in patient.structures.items()}
    clinical = patient.dose.cpu().numpy() * fractions   # load_dicom divides by fractions
    return patient.density_image, clinical, kept, masks, fractions


def check_energies(plan_path: Path, patient: str, expect_mv: float = 10.0) -> None:
    """Warn if any beam carrying MU is not at the expected energy."""
    import pydicom
    ds = pydicom.dcmread(str(plan_path), stop_before_pixels=True)
    mu = {int(r.ReferencedBeamNumber): float(getattr(r, "BeamMeterset", 0.0))
          for r in ds.FractionGroupSequence[0].ReferencedBeamSequence}
    for b in ds.BeamSequence:
        if mu.get(int(b.BeamNumber), 0.0) <= 0.0:
            continue
        e = float(b.ControlPointSequence[0].NominalBeamEnergy)
        if abs(e - expect_mv) > 1e-6:
            print(f"  WARNING {patient}: beam {b.BeamNumber} delivers "
                  f"{mu[int(b.BeamNumber)]:.1f} MU at {e:g} MV, not {expect_mv:g} MV")
