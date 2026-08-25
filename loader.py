"""Load a Vienna-cohort patient and its treatment plan for PyDoseRT.

A Vienna patient folder contains:
    CT_*.nii.gz             the planning CT, in Hounsfield units
    mask_<Struct>_*.nii.gz  binary structure masks (PTV, Rectum, Bladder, ...)
    Dose_0_*.nii.gz         the clinical ("ground-truth") total dose
    plan_*.json             the plan: per-control-point MLC leaves / jaws / gantry / MU

The CT+masks and the dose are stored on DIFFERENT voxel grids, so the first thing
we do is resample everything, in physical (mm) space, onto one common isotropic
grid. After that, CT, masks and dose all line up voxel-for-voxel.
"""

import glob
import json
import math
import os

import numpy as np
import torch
import SimpleITK as sitk

import pydosert as PDRT
from pydosert.data.beam import Beam, BeamSequence

SPACING_MM = 2.0  # the common isotropic grid we resample everything onto

# nifti structure file name  ->  the name we store the mask under
STRUCTURES = {
    "Body": "External",
    "PTV_PLOCAL": "PTV",
    "Rectum": "Rectum",
    "Bladder": "Bladder",
    "FemoralHead_L": "FemoralHead_L",
    "FemoralHead_R": "FemoralHead_R",
}


def _reference_grid(ct_img, spacing_mm):
    """A SimpleITK grid with the CT's physical extent but isotropic `spacing_mm` voxels."""
    size = [int(round(n * s / spacing_mm)) for n, s in zip(ct_img.GetSize(), ct_img.GetSpacing())]
    ref = sitk.Image(size, ct_img.GetPixelID())
    ref.SetSpacing((spacing_mm,) * 3)
    ref.SetOrigin(ct_img.GetOrigin())
    ref.SetDirection(ct_img.GetDirection())
    return ref


def _resample(img, ref, interpolator):
    """Resample `img` onto the reference grid and return a numpy array [D, H, W]."""
    out = sitk.Resample(img, ref, sitk.Transform(), interpolator, 0.0, img.GetPixelID())
    return sitk.GetArrayFromImage(out)


def load_patient(patient_dir, spacing_mm=SPACING_MM, device="cuda"):
    """Load a Vienna patient onto a common grid.

    Returns:
        patient : a pydosert Patient (holds the CT-derived density and the GT dose)
        ref     : the SimpleITK reference grid (needed later to place the plan isocentre)
        masks   : {name: bool array [D, H, W]} the structure masks
    """
    ct_path = sorted(glob.glob(os.path.join(patient_dir, "CT_*.nii.gz")))[0]
    ct_img = sitk.ReadImage(ct_path)
    ref = _reference_grid(ct_img, spacing_mm)

    ct = _resample(ct_img, ref, sitk.sitkLinear).astype(np.float32)  # Hounsfield units

    masks = {}
    for fname, name in STRUCTURES.items():
        found = sorted(glob.glob(os.path.join(patient_dir, f"mask_{fname}_*.nii.gz")))
        if found:  # nearest-neighbour keeps masks binary
            masks[name] = _resample(sitk.ReadImage(found[0]), ref, sitk.sitkNearestNeighbor) > 0.5

    dose_files = (sorted(glob.glob(os.path.join(patient_dir, "Dose_0_*.nii.gz")))
                  or sorted(glob.glob(os.path.join(patient_dir, "Dose_*.nii.gz"))))
    gt_dose = _resample(sitk.ReadImage(dose_files[0]), ref, sitk.sitkLinear).astype(np.float32)

    # pydosert's Patient converts the HU CT to physical density internally.
    patient = PDRT.Patient(
        ct_tensor=torch.from_numpy(ct),
        dose=torch.from_numpy(gt_dose),
        resolution=(spacing_mm,) * 3,
    )
    for name, mask in masks.items():
        patient.add_mask(name, torch.from_numpy(np.ascontiguousarray(mask)))

    return patient.to(device).to(torch.float32), ref, masks


def _isocentre_to_grid(iso_xyz_mm, ref, spacing_mm):
    """Map the plan's DICOM isocentre (x, y, z mm) to the engine's grid frame (z, y, x mm)."""
    ix, iy, iz = ref.TransformPhysicalPointToContinuousIndex([float(v) for v in iso_xyz_mm])
    return (iz * spacing_mm, iy * spacing_mm, ix * spacing_mm)


def load_beam_sequence(patient_dir, ref, spacing_mm=SPACING_MM, device="cuda"):
    """Parse plan_*.json into a pydosert BeamSequence (one Beam per control point).

    Returns:
        beam_sequence : the full BeamSequence (all arcs, all control points)
        fractions     : number of fractions. BeamMeterset is PER-FRACTION MU, so the
                        engine dose must be multiplied by `fractions` to get the total.
        beams         : the list of individual Beam objects (handy for per-beam demos)
    """
    plan_path = sorted(glob.glob(os.path.join(patient_dir, "plan_*.json")))[0]
    plan = json.load(open(plan_path))
    fractions = int(round(float(plan.get("Fractions", 1))))

    beams = []
    for arc in plan["BeamSequence"]:
        gantry = np.asarray(arc["GantryAngles"], dtype=np.float64)             # degrees, per CP
        cum_mu = np.asarray(arc["CumulativeMetersetWeights"], dtype=np.float64)  # 0..1, per CP
        leaves = np.asarray(arc["MLCX"], dtype=np.float32)                     # [CP, N_leaves, 2] mm
        jaws = np.asarray(arc["ASYMY"], dtype=np.float32)                      # [CP, 2] mm
        ssd = np.asarray(arc.get("SourceToSurfaceDistances", [1000.0] * len(gantry)), dtype=np.float64)
        meterset = float(arc["BeamMeterset"])                                  # per-fraction MU
        sad = float(arc["SourceAxisDistance"])
        collimator = math.radians(float(arc.get("BeamLimitingDeviceAngle", 0.0)))
        iso = _isocentre_to_grid(arc["Isocenter"], ref, spacing_mm)

        # cumulative meterset -> incremental MU delivered at each control point
        mu = np.diff(cum_mu, prepend=0.0) * meterset

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

    beam_sequence = BeamSequence.from_beams(beams).to(device).to(torch.float32)
    return beam_sequence, fractions, beams
