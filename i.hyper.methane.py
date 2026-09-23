#!/usr/bin/env python3
##############################################################################
# MODULE:    i.hyper.methane
# AUTHOR(S): Created for hyperspectral methane matched-filter mapping
# PURPOSE:   Compute a methane (CH4) matched-filter enhancement map from a
#            hyperspectral cube's SWIR bands
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Compute a methane (CH4) matched-filter enhancement map from hyperspectral SWIR bands
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
# % description: Output methane matched-filter enhancement raster
# % guisection: Output
# %end

# %option
# % key: absorption_wavelength
# % type: double
# % required: no
# % answer: 2298
# % description: Center wavelength of the CH4 absorption feature the target template is built around (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: left_shoulder
# % type: double
# % required: no
# % answer: 2120
# % description: Lower bound of the spectral window used for the matched filter (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: right_shoulder
# % type: double
# % required: no
# % answer: 2400
# % description: Upper bound of the spectral window used for the matched filter (nanometers)
# % guisection: Wavelengths
# %end

# %option
# % key: tolerance
# % type: double
# % required: no
# % answer: 15
# % description: Maximum allowed distance (nanometers) between absorption_wavelength and the nearest available band
# % guisection: Wavelengths
# %end

# %option
# % key: template_fwhm
# % type: double
# % required: no
# % answer: 40
# % description: Full-width-at-half-maximum (nanometers) of the synthetic Gaussian CH4 absorption template, used when absorption_spectrum is not given
# % guisection: Wavelengths
# %end

# %option G_OPT_F_INPUT
# % key: absorption_spectrum
# % required: no
# % description: Optional two-column text file (wavelength_nm, absorption) with a real CH4 unit absorption spectrum, used as the target template instead of the synthetic Gaussian
# % guisection: Retrieval
# %end

# %option
# % key: shrinkage
# % type: double
# % required: no
# % answer: 0.01
# % description: Diagonal loading fraction added to the background covariance matrix for numerical stability
# % guisection: Retrieval
# %end

import os
import sys
import uuid

import grass.script as gs

MIN_WINDOW_BANDS = 5
MIN_VALID_PIXELS_PER_BAND = 5


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
            "absorption region -- i.hyper.methane's matched filter requires bands "
            "near 2100-2400 nm."
        )
    gs.verbose(f"{label}: requested {target_wl} nm -> band {closest['band_num']} at {closest['wavelength']:.1f} nm")
    return closest


def select_window_bands(bands, wl_left: float, wl_right: float):
    return [b for b in bands if wl_left <= b["wavelength"] <= wl_right]


def build_absorption_template(wavelengths_nm, absorption_wavelength, template_fwhm, absorption_spectrum_path):
    """Return a positive-valued absorption-strength template (peak = 1) sampled
    at `wavelengths_nm`, either interpolated from a user-supplied unit
    absorption spectrum file, or a synthetic Gaussian centered on
    absorption_wavelength (a simplified stand-in for a real HITRAN-derived
    CH4 cross-section curve)."""
    import numpy as np

    if absorption_spectrum_path:
        try:
            data = np.loadtxt(absorption_spectrum_path, comments="#")
        except Exception as exc:
            gs.fatal(f"Cannot read absorption_spectrum file <{absorption_spectrum_path}>: {exc}")
        if data.ndim != 2 or data.shape[1] < 2:
            gs.fatal(
                f"absorption_spectrum file <{absorption_spectrum_path}> must have two "
                "whitespace/comma-separated columns: wavelength_nm, absorption"
            )
        file_wl, file_abs = data[:, 0], data[:, 1]
        order = np.argsort(file_wl)
        file_wl, file_abs = file_wl[order], np.abs(file_abs[order])
        if wavelengths_nm.min() < file_wl.min() or wavelengths_nm.max() > file_wl.max():
            gs.fatal(
                f"absorption_spectrum file <{absorption_spectrum_path}> covers "
                f"{file_wl.min():.1f}-{file_wl.max():.1f} nm, which does not span the "
                f"requested matched-filter window {wavelengths_nm.min():.1f}-"
                f"{wavelengths_nm.max():.1f} nm."
            )
        template = np.interp(wavelengths_nm, file_wl, file_abs)
    else:
        sigma = template_fwhm / 2.3548200450309493  # FWHM -> Gaussian sigma
        template = np.exp(-0.5 * ((wavelengths_nm - absorption_wavelength) / sigma) ** 2)

    peak = template.max()
    if not np.isfinite(peak) or peak <= 0:
        gs.fatal(
            "The absorption template is degenerate (all zero or non-finite). Check "
            "absorption_wavelength/template_fwhm, or the absorption_spectrum file contents."
        )
    return template / peak


def main():
    options, flags = gs.parser()

    input3d = options["input"]
    output = options["output"]
    wl_absorption = float(options["absorption_wavelength"])
    wl_left = float(options["left_shoulder"])
    wl_right = float(options["right_shoulder"])
    tolerance = float(options["tolerance"])
    template_fwhm = float(options["template_fwhm"])
    absorption_spectrum_path = options["absorption_spectrum"] or None
    shrinkage = float(options["shrinkage"])

    if not (wl_left < wl_absorption < wl_right):
        gs.fatal(
            f"left_shoulder ({wl_left}) < absorption_wavelength ({wl_absorption}) < "
            f"right_shoulder ({wl_right}) must hold."
        )

    gs.message(
        "NOTE: this is a matched-filter enhancement statistic (background-covariance-"
        "whitened correlation with a CH4 absorption template), not a radiometrically "
        "calibrated ppm-m column retrieval, unless absorption_spectrum supplies a "
        "properly scaled unit absorption spectrum. Use it to rank and flag candidate "
        "hotspots."
    )

    # Lazy-import: only needed once we actually run the retrieval, not for --help/--interface-description.
    import numpy as np
    from grass.script import array as garray

    bands = get_all_band_wavelengths(input3d)
    nearest_band(bands, wl_absorption, tolerance, "absorption_wavelength")
    window_bands = select_window_bands(bands, wl_left, wl_right)
    if len(window_bands) < MIN_WINDOW_BANDS:
        gs.fatal(
            f"Only {len(window_bands)} band(s) of <{input3d}> fall within the "
            f"{wl_left}-{wl_right} nm matched-filter window; need at least "
            f"{MIN_WINDOW_BANDS} to estimate a background covariance matrix. This "
            "input's SWIR coverage or spectral sampling may not be sufficient for "
            "matched-filter methane retrieval."
        )

    tmp_prefix = gs.tempname(8)
    band_maps = [f"{tmp_prefix}_b{i}" for i in range(len(window_bands))]
    try:
        for name2d, band in zip(band_maps, window_bands):
            extract_z_slice(input3d, band["band_num"] - 1, name2d)

        stack = np.stack(
            [np.array(garray.array(mapname=m, dtype=np.float64, null=np.nan)) for m in band_maps],
            axis=0,
        )
    finally:
        gs.run_command("g.remove", type="raster", name=",".join(band_maps), flags="f", quiet=True)

    nbands, nrows, ncols = stack.shape
    flat = stack.reshape(nbands, -1)
    valid_mask = np.all(np.isfinite(flat), axis=0)
    n_valid = int(valid_mask.sum())
    if n_valid < nbands * MIN_VALID_PIXELS_PER_BAND:
        gs.fatal(
            f"Only {n_valid} pixels have finite values across all {nbands} matched-filter "
            f"bands; need at least {nbands * MIN_VALID_PIXELS_PER_BAND} to estimate a "
            "stable background covariance matrix. Check the input's null/coverage or "
            "widen the computational region."
        )

    x_valid = flat[:, valid_mask]
    mu = x_valid.mean(axis=1)
    centered = x_valid - mu[:, None]
    cov = np.cov(centered)
    cov += shrinkage * np.mean(np.diag(cov)) * np.eye(nbands)

    wavelengths_nm = np.array([b["wavelength"] for b in window_bands])
    template = build_absorption_template(wavelengths_nm, wl_absorption, template_fwhm, absorption_spectrum_path)
    # Linearized optically-thin absorption: dRadiance/dAlpha ~= -mu * template.
    target = -mu * template

    try:
        weights = np.linalg.solve(cov, target)
    except np.linalg.LinAlgError as exc:
        gs.fatal(f"Background covariance matrix is singular and cannot be inverted: {exc}")

    denom = float(target @ weights)
    if not np.isfinite(denom) or denom <= 0:
        gs.fatal(
            "Matched-filter denominator is non-positive or non-finite; the background "
            "covariance may be degenerate (e.g. too few valid pixels, or a near-constant "
            "scene). Try widening the region, adjusting the spectral window, or "
            "increasing shrinkage."
        )

    alpha_valid = (centered.T @ weights) / denom

    alpha_flat = np.full(nrows * ncols, np.nan, dtype=np.float64)
    alpha_flat[valid_mask] = alpha_valid
    alpha_2d = alpha_flat.reshape(nrows, ncols)

    out_arr = garray.array()
    out_arr[...] = alpha_2d
    out_arr.write(mapname=output, overwrite=gs.overwrite(), quiet=True)

    template_source = absorption_spectrum_path if absorption_spectrum_path else (
        f"synthetic Gaussian, center {wl_absorption} nm, FWHM {template_fwhm} nm"
    )
    gs.raster_history(output, overwrite=True)
    gs.run_command(
        "r.support", map=output,
        title="Methane matched-filter enhancement",
        description=(
            f"Matched-filter CH4 enhancement from <{input3d}>: {nbands} bands over "
            f"{wl_left:.1f}-{wl_right:.1f} nm, target template: {template_source}, "
            f"shrinkage {shrinkage}. Relative hotspot-ranking statistic, not a "
            "calibrated ppm-m retrieval unless absorption_spectrum is radiometrically scaled."
        ),
    )
    gs.message(f"Wrote methane matched-filter enhancement to <{output}>")


if __name__ == "__main__":
    main()
