#!/usr/bin/env python3
##############################################################################
# MODULE:    plume_wind (internal helper for i.hyper.methane)
# AUTHOR(S): Created for the i.hyper module family, ALMA Energy project
# PURPOSE:   Plume direction (resolved from the image itself, anchored at a
#            known source -- no wind data required) plus an IME-based flux
#            estimate for i.hyper.methane hotspots that DOES need ERA5 wind
#            speed (via t.in.era5), since converting a spatial mass
#            enhancement into an emission RATE fundamentally requires a
#            transport velocity -- direction and speed are two separate
#            problems, and only speed needs wind data. Wind direction, when
#            fetched, is reported only as an informational cross-check
#            against the image-derived direction (Varon et al. 2018's
#            cross-sectional-flux formulation; the wind-alignment idea
#            traces to Valverde et al. 2024, but here the image is trusted
#            over reanalysis for direction -- see NOTES).
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Image-derived plume direction and IME-based flux estimate (ERA5 wind speed via t.in.era5) for an i.hyper.methane hotspot
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: methane
# % keyword: temporal
# %end

# %option G_OPT_R_INPUT
# % key: alpha_map
# % required: yes
# % description: i.hyper.methane matched-filter output raster
# % guisection: Input
# %end

# %option G_OPT_M_COORDS
# % key: source
# % required: yes
# % description: Easting,northing of the candidate source/hotspot, in the current project's CRS
# % guisection: Input
# %end

# %flag
# % key: i
# % description: Image direction only -- skip ERA5 entirely (no date/hour needed, no wind comparison, no flux). Use when you only want the direction resolved from the plume's own spatial pattern, not an emission-rate estimate
# % guisection: Input
# %end

# %option
# % key: date
# % type: string
# % required: no
# % description: Acquisition date, YYYY-MM-DD, for the ERA5 wind lookup. Required unless -i is given
# % guisection: Input
# %end

# %option
# % key: hour
# % type: string
# % required: no
# % description: Acquisition hour, HH:MM:SS (UTC), to fetch hourly ERA5-Land wind and sample the specific overpass hour instead of the daily mean -- the daily mean can substantially smooth short-lived wind conditions relevant to a specific plume observation
# % guisection: Input
# %end

# %option
# % key: radius
# % type: double
# % required: no
# % answer: 500
# % description: Radius (map units) around source to consider for plume-orientation and flux estimation
# % guisection: Input
# %end

# %option
# % key: percentile
# % type: double
# % required: no
# % answer: 90
# % description: Percentile threshold (of alpha_map values within radius) above which pixels count as plume for orientation/flux
# % guisection: Input
# %end

# %option
# % key: consistency_tolerance
# % type: double
# % required: no
# % answer: 45
# % description: Maximum angle (degrees) between plume orientation and wind direction to call them consistent
# % guisection: Retrieval
# %end

# %option
# % key: alpha_to_ppm_m
# % type: double
# % required: no
# % description: Conversion factor from alpha_map units to ppm-m CH4 enhancement. Required for an absolute flux (kg/h) estimate -- with i.hyper.methane's default synthetic-Gaussian target this factor is not known, so omitting it reports a relative flux proxy only
# % guisection: Retrieval
# %end

# %option
# % key: output_prefix
# % type: string
# % required: no
# % answer: plume_wind
# % description: STRDS name prefix for the ERA5 wind data (passed to t.in.era5)
# % guisection: Retrieval
# %end

# %option
# % key: era5_cache_dir
# % type: string
# % required: no
# % description: Cache directory for t.in.era5 downloads (default: a run-scoped temporary directory, re-downloaded every run)
# % guisection: Retrieval
# %end

import math

import grass.script as gs

# CH4 molecular weight (g/mol) and Avogadro's number, for ppm-m -> kg/m^2.
CH4_MOLAR_MASS_G = 16.04
AVOGADRO = 6.02214076e23
# Sea-level number density of air (Loschmidt-like reference), molecules/cm^3,
# used to convert a ppm-m path amount into a molecular column density.
N0_AIR_CM3 = 2.546e19


def fetch_wind(date, output_prefix, cache_dir=None, hourly=False):
    """Fetch ERA5 10 m wind_u/wind_v (via t.in.era5) over the current GRASS
    region's extent (reprojected to WGS84 automatically by g.region -bg) for
    the given date. Returns the (strds_u, strds_v) names."""
    reg = gs.parse_command("g.region", flags="bg")
    area = f"{reg['ll_n']},{reg['ll_w']},{reg['ll_s']},{reg['ll_e']}"
    kwargs = dict(
        variables="wind_u,wind_v", start=date, end=date,
        area=area, output_prefix=output_prefix,
    )
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    gs.run_command("t.in.era5", flags="h" if hourly else "", quiet=True, **kwargs)
    return f"{output_prefix}_wind_u", f"{output_prefix}_wind_v"


def _last_value(t_rast_what_output):
    lines = [ln for ln in t_rast_what_output.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    val = lines[-1].split("|")[-1]
    if val in ("", "*"):
        return None
    return float(val)


def sample_wind_at_point(strds_u, strds_v, easting, northing, at_time=None):
    """Sample U/V 10 m wind components (m/s) at a point (current project's
    native CRS) from the given STRDS. Returns (speed_m_s, direction_from_deg)
    where direction_from_deg is the meteorological convention (direction the
    wind blows FROM: 0=N, 90=E, ...), or (None, None) if no data is found.

    `at_time`, if given, is a "YYYY-MM-DD HH:MM:SS" string selecting one
    specific timestep out of a strds fetched with fetch_wind(..., hourly=True)
    -- without it, an hourly strds has 24 rows and this would silently pick
    whichever one t.rast.what happens to list last (an arbitrary hour, not
    necessarily the one an overpass/plume actually occurred at)."""
    coord_str = f"{easting},{northing}"
    where = f"start_time = '{at_time}'" if at_time else None
    out_u = gs.read_command(
        "t.rast.what", strds=strds_u, coordinates=coord_str, flags="n",
        separator="pipe", where=where,
    )
    out_v = gs.read_command(
        "t.rast.what", strds=strds_v, coordinates=coord_str, flags="n",
        separator="pipe", where=where,
    )
    u, v = _last_value(out_u), _last_value(out_v)
    if u is None or v is None:
        return None, None
    speed = math.hypot(u, v)
    direction_from = math.degrees(math.atan2(-u, -v)) % 360.0
    return speed, direction_from


def plume_pixels_near_source(alpha_map, easting, northing, radius, percentile):
    """Return (x_offsets, y_offsets, values) for alpha_map pixels within
    `radius` map units of (easting, northing) whose value exceeds the given
    percentile of the values in that same neighborhood."""
    import numpy as np
    from grass.script import array as garray

    reg = gs.region()
    arr = np.array(garray.array(mapname=alpha_map, dtype=np.float64, null=np.nan))
    nrows, ncols = arr.shape
    xres = (reg["e"] - reg["w"]) / ncols
    yres = (reg["n"] - reg["s"]) / nrows
    col_idx = np.arange(ncols)
    row_idx = np.arange(nrows)
    xs = reg["w"] + (col_idx + 0.5) * xres
    ys = reg["n"] - (row_idx + 0.5) * yres
    xx, yy = np.meshgrid(xs, ys)

    dist = np.hypot(xx - easting, yy - northing)
    near = (dist <= radius) & np.isfinite(arr)
    if not near.any():
        return None, None, None
    threshold = np.nanpercentile(arr[near], percentile)
    # Strict ">": see the matching comment in plume_artifact_check.py --
    # avoids including a tied low-value plateau when it spans the percentile.
    # Falls back to ">=" if that leaves nothing (a tied high plateau instead).
    plume = near & (arr > threshold)
    if not plume.any():
        plume = near & (arr >= threshold)
    if plume.sum() < 3:
        return None, None, None
    return xx[plume] - easting, yy[plume] - northing, arr[plume]


def plume_orientation_deg(x_offsets, y_offsets):
    """Principal orientation (degrees from north, mod 180 -- an elongated
    cluster has no inherent sign) of a set of point offsets, via the 2nd
    moment / covariance matrix (equivalent to PCA on the point cloud)."""
    import numpy as np

    cov = np.cov(np.vstack([x_offsets, y_offsets]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    angle_from_north = math.degrees(math.atan2(principal[0], principal[1])) % 180.0
    return angle_from_north


def plume_direction_deg(x_offsets, y_offsets, values):
    """Signed downwind direction (0-360 degrees from north, the direction
    the plume extends TOWARD) resolved directly from the image, anchored at
    a known source location -- no wind data needed for this.

    Uses the same principal axis as plume_orientation_deg(), then resolves
    *which* of the two directions along that axis is downwind via the
    value-weighted centroid of the hotspot pixels: since x_offsets/
    y_offsets are already relative to the source (0,0; see
    plume_pixels_near_source()), a plume that extends away from the source
    pulls the weighted centroid off to one side of it, and that side is
    downwind -- the source itself is (by construction) the near-zero-
    concentration end, not the peak.

    Returns None if the weighted centroid sits essentially on the source
    (no resolvable directional signal in the image alone, e.g. a
    symmetric/point-like hotspot) -- callers should fall back to
    plume_orientation_deg()'s unsigned axis in that case."""
    import numpy as np

    x = np.asarray(x_offsets, dtype=np.float64)
    y = np.asarray(y_offsets, dtype=np.float64)
    w = np.asarray(values, dtype=np.float64)

    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]

    w_sum = np.sum(w)
    if w_sum <= 0:
        return None
    centroid = np.array([np.sum(w * x), np.sum(w * y)]) / w_sum
    projection = float(np.dot(centroid, axis))

    # Degeneracy test: is the weighted centroid's offset along the axis
    # larger than its own sampling noise, or indistinguishable from the
    # zero-offset (symmetric-around-source) case for this many/these
    # points? A finite sample from a truly symmetric distribution has a
    # nonzero centroid by chance (~O(1/sqrt(n_eff))), so comparing the
    # projection to an absolute epsilon would flag those as "resolved"
    # incorrectly -- compare it to its own weighted standard error instead.
    proj_i = x * axis[0] + y * axis[1]
    weighted_var = np.sum(w * (proj_i - projection) ** 2) / w_sum
    n_eff = (w_sum**2) / np.sum(w**2)  # Kish effective sample size
    std_err = math.sqrt(weighted_var / max(n_eff, 1.0))
    if std_err <= 0 or abs(projection) < 2.0 * std_err:
        return None

    if projection < 0:
        axis = -axis
    return math.degrees(math.atan2(axis[0], axis[1])) % 360.0


def wind_consistency(plume_angle_deg, wind_from_deg, tolerance_deg):
    """Angle (degrees, 0-90) between the plume's (unsigned) elongation axis
    and the wind's downwind axis, and whether it is within tolerance_deg.
    Use direction_agreement() instead when plume_direction_deg() resolved a
    signed direction -- this mod-180 comparison is only the fallback for
    when it could not."""
    wind_to_deg = (wind_from_deg + 180.0) % 360.0
    a, b = plume_angle_deg % 180.0, wind_to_deg % 180.0
    diff = abs(a - b)
    diff = min(diff, 180.0 - diff)
    return diff, diff <= tolerance_deg


def direction_agreement(direction_a_deg, direction_b_deg, tolerance_deg):
    """Angular difference (0-180 deg) between two full 0-360 'extends/blows
    toward' directions, and whether it's within tolerance_deg. A mismatch
    here is informational, not disqualifying -- it can reflect genuine
    local wind effects a coarse reanalysis grid cell doesn't resolve (see
    plume_wind.py's own NOTES for a real example), not necessarily an error
    in either estimate."""
    diff = abs(direction_a_deg - direction_b_deg) % 360.0
    diff = min(diff, 360.0 - diff)
    return diff, diff <= tolerance_deg


def ppm_m_to_kg_m2(ppm_m):
    """Convert a CH4 path-column enhancement in ppm-m to a mass column
    density in kg/m^2, via a Loschmidt-referenced sea-level air number
    density (this is the standard back-of-envelope conversion used across
    the point-source methane matched-filter literature, e.g. Thorpe et al.
    2013 / Frankenberg et al. 2016 -- not a full radiative-transfer-derived
    air-mass-factor column, which would additionally depend on scene
    altitude/pressure)."""
    n_cm2_per_ppmm = N0_AIR_CM3 * 1e-6 * 100.0  # molecules/cm^2 per 1 ppm-m
    n_cm2 = ppm_m * n_cm2_per_ppmm
    n_m2 = n_cm2 * 1e4
    kg_per_molecule = (CH4_MOLAR_MASS_G / AVOGADRO) / 1000.0
    return n_m2 * kg_per_molecule


def estimate_flux(values, pixel_area_m2, plume_area_m2, wind_speed_ms, alpha_to_ppm_m):
    """IME-based point-source flux (Varon et al. 2018): Q = U_eff * IME / L,
    with IME the integrated mass enhancement over the plume mask and L the
    plume's effective length scale (sqrt of its area). U_eff is taken as the
    10 m wind speed directly (a simplification -- Varon et al. found
    U_eff/U10 ratios of roughly 0.3-0.5 to 1 depending on plume-detection
    algorithm and wind regime; passing a pre-scaled wind_speed_ms accounts
    for that if desired).

    Returns Q in kg/h, or None (with values still usable as a *relative*
    proxy) if alpha_to_ppm_m is not supplied, since i.hyper.methane's alpha
    is not radiometrically calibrated by default."""
    if alpha_to_ppm_m is None:
        return None
    ppm_m_values = values * alpha_to_ppm_m
    ime_kg = sum(max(v, 0.0) * ppm_m_to_kg_m2(1.0) * pixel_area_m2 for v in ppm_m_values)
    length_scale_m = math.sqrt(plume_area_m2)
    if length_scale_m <= 0:
        return None
    q_kg_s = wind_speed_ms * ime_kg / length_scale_m
    return q_kg_s * 3600.0


def main():
    options, flags = gs.parser()

    alpha_map = options["alpha_map"]
    easting, northing = (float(v) for v in options["source"].split(","))
    image_only = bool(flags["i"])
    date = options["date"] or None
    hour = options["hour"] or None
    radius = float(options["radius"])
    percentile = float(options["percentile"])
    tolerance = float(options["consistency_tolerance"])
    alpha_to_ppm_m = float(options["alpha_to_ppm_m"]) if options["alpha_to_ppm_m"] else None
    output_prefix = options["output_prefix"]
    cache_dir = options["era5_cache_dir"] or None

    if not image_only and not date:
        gs.fatal("date= is required unless -i (image direction only) is given.")

    import numpy as np

    x_off, y_off, values = plume_pixels_near_source(alpha_map, easting, northing, radius, percentile)
    if x_off is None:
        gs.fatal(
            f"No pixels above the {percentile}th percentile found within {radius} map "
            f"units of ({easting}, {northing}) in <{alpha_map}>."
        )

    # Direction, resolved from the image alone, anchored at the known source
    # -- no wind data needed for this part at all.
    direction = plume_direction_deg(x_off, y_off, values)
    if direction is not None:
        direction_note = "resolved from the image (source-anchored, value-weighted centroid)"
    else:
        direction = plume_orientation_deg(x_off, y_off)
        direction_note = (
            "UNSIGNED axis only (mod 180 deg) -- the hotspot's weighted centroid sits "
            "essentially on the source, so the image alone cannot resolve which end is downwind"
        )
    gs.message(f"Plume direction: {direction:.1f} deg from north ({direction_note}).")

    if image_only:
        gs.message("(-i given: skipping ERA5 wind fetch and flux estimate entirely.)")
        return

    strds_u, strds_v = fetch_wind(date, output_prefix, cache_dir=cache_dir, hourly=bool(hour))
    at_time = f"{date} {hour}" if hour else None
    wind_speed, wind_from = sample_wind_at_point(strds_u, strds_v, easting, northing, at_time=at_time)
    if wind_speed is None:
        gs.fatal(
            f"Could not sample ERA5 wind at ({easting}, {northing}) on {date}"
            + (f" {hour}" if hour else "") + "."
        )
    wind_to = (wind_from + 180.0) % 360.0

    # Direction agreement is reported for awareness only -- it does NOT gate
    # anything below. A mismatch can reflect genuine local wind effects a
    # ~9-31 km reanalysis grid cell doesn't resolve (observed in practice:
    # ERA5-Land gave a downwind direction ~180 deg from the true wind at a
    # real published plume site, in light-wind/complex-terrain conditions),
    # not necessarily an error in the image-derived direction.
    if direction is not None:
        angle_diff, agrees = direction_agreement(direction, wind_to, tolerance)
    else:
        angle_diff, agrees = wind_consistency(direction, wind_from, tolerance)

    reg = gs.region()
    pixel_area_m2 = ((reg["e"] - reg["w"]) / reg["cols"]) * ((reg["n"] - reg["s"]) / reg["rows"])
    plume_area_m2 = pixel_area_m2 * len(values)

    flux_kg_h = estimate_flux(values, pixel_area_m2, plume_area_m2, wind_speed, alpha_to_ppm_m)

    gs.message(
        f"ERA5 wind at ({easting}, {northing}) on {date}"
        f"{' ' + hour + ' UTC (ERA5-Land hourly)' if hour else ' (ERA5-Land/ERA5 daily mean)'}: "
        f"{wind_speed:.2f} m/s, from {wind_from:.1f} deg (blowing toward {wind_to:.1f} deg).\n"
        f"Image vs. ERA5 downwind direction: {angle_diff:.1f} deg apart "
        f"({'agree' if agrees else 'DISAGREE'} within tolerance {tolerance} deg) -- "
        "informational only; only the wind SPEED feeds the flux estimate below, "
        "not this direction comparison."
    )
    if flux_kg_h is not None:
        gs.message(f"Estimated flux: {flux_kg_h:.1f} kg/h (IME method, calibrated via alpha_to_ppm_m={alpha_to_ppm_m}).")
    else:
        relative_ime = float(np.sum(np.clip(values, 0, None)))
        gs.message(
            f"No alpha_to_ppm_m given -- flux not reported in kg/h. Relative integrated "
            f"enhancement over the plume mask: {relative_ime:.4g} (alpha_map units x pixel count); "
            "only comparable across runs of the same calibration, not to literature kg/h values."
        )


if __name__ == "__main__":
    main()
