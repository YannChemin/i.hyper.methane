#!/usr/bin/env python3
##############################################################################
# MODULE:    i.hyper.methane
# AUTHOR(S): Created for hyperspectral methane band-depth mapping
# PURPOSE:   Compute a methane absorption band-depth proxy index from a
#            hyperspectral cube's SWIR bands
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Compute a methane (CH4) absorption band-depth proxy index from hyperspectral SWIR bands
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: methane
# % keyword: spectral analysis
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % required: yes
# % description: Input hyperspectral 3D raster map (from i.hyper.import), covering the SWIR methane absorption region
# % guisection: Input
# %end

# %option G_OPT_R_OUTPUT
# % key: output
# % required: yes
# % description: Output methane band-depth index raster
# % guisection: Output
# %end

# %option
# % key: absorption_wavelength
# % type: double
# % required: no
# % answer: 2298
# % description: Center wavelength of the CH4 absorption feature to use (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: left_shoulder
# % type: double
# % required: no
# % answer: 2120
# % description: Continuum reference wavelength shortward of the absorption feature (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: right_shoulder
# % type: double
# % required: no
# % answer: 2400
# % description: Continuum reference wavelength longward of the absorption feature (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: tolerance
# % type: double
# % required: no
# % answer: 15
# % description: Maximum allowed distance (nanometers) between a requested wavelength and the nearest available band
# % guisection: Wavelengths
# %end

import os
import sys
import uuid

import grass.script as gs


def extract_z_slice(name3d: str, z: int, name2d: str) -> None:
    """Extract 0-based Z-slice z from `name3d` into 2D raster `name2d`,
    via r3.cross.rast against a constant-elevation raster at the slice's
    center Z coordinate (stock GRASS, no private extensions needed)."""
    region3 = gs.parse_command("g.region", flags="3g")
    bottom = float(region3["b"])
    tbres = float(region3["tbres"])
    z_center = bottom + (z + 0.5) * tbres

    elev_tmp = f"{name2d}_elev_{uuid.uuid4().hex[:8]}"
    gs.mapcalc(f"{elev_tmp} = {z_center}", overwrite=True, quiet=True)
    try:
        gs.run_command(
            "r3.cross.rast", input=name3d, elevation=elev_tmp, output=name2d,
            overwrite=True, quiet=True,
        )
    finally:
        gs.run_command("g.remove", type="raster", name=elev_tmp, flags="f", quiet=True)


def parse_wavelength_from_metadata(raster3d: str, band_num: int):
    band_name = f"{raster3d}#{band_num}"
    wavelength, unit = None, "nm"
    try:
        result = gs.read_command("r.support", map=band_name, flags="n")
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("wavelength="):
                wavelength = float(line.split("=")[1])
            elif line.startswith("unit="):
                unit = line.split("=")[1].strip()
    except Exception:
        pass
    return wavelength, unit


def convert_wavelength_to_nm(wavelength: float, unit: str) -> float:
    unit = unit.lower().strip()
    if unit in ("nm", "nanometer", "nanometers"):
        return wavelength
    if unit in ("um", "µm", "micrometer", "micrometers", "micron", "microns"):
        return wavelength * 1000.0
    if unit in ("m", "meter", "meters"):
        return wavelength * 1e9
    gs.warning(f"Unknown wavelength unit '{unit}', assuming nanometers")
    return wavelength


def _load_hyper_json_bands(raster3d: str):
    """Read wavelength metadata from i.hyper.import's JSON sidecar, if present.

    i.hyper.import stores per-band wavelengths at
    $MAPSET/grid3/<mapname>/hyper.json rather than in r3.support history.
    """
    import json

    name, mapset = (raster3d.split("@", 1) if "@" in raster3d else (raster3d, None))
    try:
        env = gs.gisenv()
        mapset = mapset or env["MAPSET"]
        path = os.path.join(env["GISDBASE"], env["LOCATION_NAME"], mapset, "grid3", name, "hyper.json")
    except Exception:
        return []
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        data = json.load(f)
    wavelengths = (data.get("bands") or {}).get("wavelength")
    if not wavelengths:
        return []
    return [{"band_num": i + 1, "wavelength": float(wl)} for i, wl in enumerate(wavelengths)]


def get_all_band_wavelengths(raster3d: str):
    """Return all bands' {band_num, wavelength} sorted by wavelength, failing
    loudly if no wavelength metadata is available anywhere."""
    json_bands = _load_hyper_json_bands(raster3d)
    if json_bands:
        json_bands.sort(key=lambda b: b["wavelength"])
        return json_bands

    try:
        info = gs.raster3d_info(raster3d)
        depths = int(info["depths"])
    except Exception as exc:
        gs.fatal(f"Cannot get info for 3D raster <{raster3d}>: {exc}")

    bands = []
    for i in range(1, depths + 1):
        wavelength, unit = parse_wavelength_from_metadata(raster3d, i)
        if wavelength is not None:
            bands.append({"band_num": i, "wavelength": convert_wavelength_to_nm(wavelength, unit)})
    if not bands:
        gs.fatal(
            f"No wavelength metadata found on <{raster3d}>'s bands. "
            "Use data imported with i.hyper.import, or add wavelength metadata via r.support."
        )
    bands.sort(key=lambda b: b["wavelength"])
    return bands


def nearest_band(bands, target_wl: float, tolerance: float, label: str):
    closest = min(bands, key=lambda b: abs(b["wavelength"] - target_wl))
    delta = abs(closest["wavelength"] - target_wl)
    if delta > tolerance:
        gs.fatal(
            f"No band within {tolerance} nm of the requested {label} wavelength "
            f"({target_wl} nm); nearest available band is {closest['wavelength']:.1f} nm "
            f"({delta:.1f} nm away). This input's SWIR coverage may not reach the CH4 "
            "absorption region -- i.hyper.methane's band-depth index requires bands "
            "near 2100-2400 nm."
        )
    gs.verbose(f"{label}: requested {target_wl} nm -> band {closest['band_num']} at {closest['wavelength']:.1f} nm")
    return closest


def main():
    options, flags = gs.parser()

    input3d = options["input"]
    output = options["output"]
    wl_absorption = float(options["absorption_wavelength"])
    wl_left = float(options["left_shoulder"])
    wl_right = float(options["right_shoulder"])
    tolerance = float(options["tolerance"])

    if not (wl_left < wl_absorption < wl_right):
        gs.fatal(
            f"left_shoulder ({wl_left}) < absorption_wavelength ({wl_absorption}) < "
            f"right_shoulder ({wl_right}) must hold."
        )

    gs.message(
        "NOTE: this is a band-depth absorption proxy (continuum-interpolated "
        "reflectance vs. an absorption band), not a calibrated ppm-m methane "
        "column retrieval. A true quantitative retrieval requires a matched-filter "
        "against a unit absorption spectrum with a scene background covariance "
        "model, which this module does not implement."
    )

    bands = get_all_band_wavelengths(input3d)
    left_band = nearest_band(bands, wl_left, tolerance, "left_shoulder")
    abs_band = nearest_band(bands, wl_absorption, tolerance, "absorption_wavelength")
    right_band = nearest_band(bands, wl_right, tolerance, "right_shoulder")

    tmp_prefix = gs.tempname(8)
    tmp_left = f"{tmp_prefix}_left"
    tmp_abs = f"{tmp_prefix}_abs"
    tmp_right = f"{tmp_prefix}_right"
    tmp_continuum = f"{tmp_prefix}_continuum"

    try:
        extract_z_slice(input3d, left_band["band_num"] - 1, tmp_left)
        extract_z_slice(input3d, abs_band["band_num"] - 1, tmp_abs)
        extract_z_slice(input3d, right_band["band_num"] - 1, tmp_right)

        frac = (wl_absorption - left_band["wavelength"]) / (right_band["wavelength"] - left_band["wavelength"])
        gs.mapcalc(
            f"{tmp_continuum} = {tmp_left} + ({tmp_right} - {tmp_left}) * {frac}",
            overwrite=True,
        )
        gs.mapcalc(
            f"{output} = if({tmp_continuum} == 0, null(), 1.0 - ({tmp_abs} / {tmp_continuum}))",
            overwrite=gs.overwrite(),
        )
    finally:
        gs.run_command(
            "g.remove", type="raster", name=f"{tmp_left},{tmp_abs},{tmp_right},{tmp_continuum}",
            flags="f", quiet=True,
        )

    gs.raster_history(output, overwrite=True)
    gs.run_command(
        "r.support", map=output,
        title="Methane absorption band-depth proxy index",
        description=(
            f"Band-depth index from <{input3d}>: absorption band {abs_band['wavelength']:.1f} nm, "
            f"continuum shoulders {left_band['wavelength']:.1f}/{right_band['wavelength']:.1f} nm. "
            "Relative hotspot-flagging proxy, not a calibrated ppm-m retrieval."
        ),
    )
    gs.message(f"Wrote methane band-depth index to <{output}>")


if __name__ == "__main__":
    main()
