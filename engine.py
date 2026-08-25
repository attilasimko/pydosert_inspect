"""Build the PyDoseRT dose engine and compute dose for a Vienna patient.

The engine takes a beam sequence + a density image and returns a 3D dose. We use the
Elekta Agility machine preset (80 MLC leaf pairs, 5 mm), which matches the Vienna plans.
"""

import torch

import pydosert as PDRT
from pydosert.data import MachineConfig
from pydosert.data.beam import BeamSequence


def build_engine(patient, beam_sequence, kernel_size=25, beam_chunk_size=8,
                 machine="elekta_10MV", device="cuda"):
    """A DoseEngine sized to the patient grid and configured for the given machine.

    kernel_size    : pencil-beam kernel width; larger = more accurate scatter, slower.
    beam_chunk_size : control points processed at once (lower = less GPU memory).
    """
    device = torch.device(device)   # the engine needs a torch.device, not a plain string
    return PDRT.DoseEngine(
        machine_config=MachineConfig(preset=machine),
        dose_grid_spacing=patient.resolution,
        dose_grid_shape=patient.density_image.shape,
        beam_template=beam_sequence,
        auto_calibrate=True,          # scale output so 1 MU gives the machine's reference dose
        kernel_size=kernel_size,
        dtype=torch.float16,          # fp16 is plenty for dose and saves memory
        device=device,
        beam_chunk_size=beam_chunk_size,
    )


def compute_dose(engine, beam_sequence, patient, fractions):
    """Recompute the full plan dose (all control points) as a TOTAL dose in Gy [D, H, W]."""
    with torch.no_grad():
        per_fraction = engine.compute_dose(beam_sequence, density_image=patient.density_image)[0]
    return (per_fraction.float() * fractions).cpu().numpy()


def compute_intermediates(engine, beam, patient):
    """Run a SINGLE control point with return_intermediates=True and return every stage of
    the dose pipeline, so you can see how PyDoseRT turns leaf/jaw positions into dose:

        fluence_map        [H, W]      the 2D opening let through the MLC + jaws
        fluence_volume     [D, H, W]   that fluence projected into the patient (beam's-eye view)
        radiological_depth [D, H, W]   water-equivalent depth along each ray (drives attenuation)
        beam_dose          [D, H, W]   the dose from this one beam

    (return_intermediates processes all beams at once, so we pass a single Beam.)
    """
    # Wrap the one beam in its own sequence and move it onto the patient's device/dtype.
    single = BeamSequence.from_beams([beam]).to(patient.density_image.device).to(torch.float32)
    with torch.no_grad():
        rad_depth, fluence_map, fluence_vol, dose = engine.compute_dose(
            single, density_image=patient.density_image, return_intermediates=True, overwrite=True)

    to_np = lambda t: t.float().cpu().numpy()[0]  # drop the beam/batch dimension
    return {
        "fluence_map": to_np(fluence_map),
        "fluence_volume": to_np(fluence_vol),
        "radiological_depth": to_np(rad_depth),
        "beam_dose": to_np(dose),
    }
