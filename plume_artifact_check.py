#!/usr/bin/env python3
##############################################################################
# MODULE:    plume_artifact_check (internal helper for i.hyper.methane)
# AUTHOR(S): Created for the i.hyper module family, ALMA Energy project
# PURPOSE:   Flag i.hyper.methane hotspots that are likely POME-pond
#            surface-reflectance retrieval artifacts rather than genuine
#            methane plumes, by comparing the hotspot's spatial footprint
#            against a Sentinel-2-derived water/pond classification --
#            the same boundary-matching check Valverde et al. 2024 (Fig. 4)
#            did by hand with NDWI, here made quantitative and reusable.
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Flag i.hyper.methane hotspots that spatially coincide with POME ponds (Sentinel-2 water classification), a signature of retrieval artifact rather than genuine emission
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: methane
# % keyword: sentinel
# %end

# %option G_OPT_R_INPUT
# % key: alpha_map
# % required: yes
# % description: i.hyper.methane matched-filter output raster
# % guisection: Input
# %end

# %option G_OPT_R_INPUT
# % key: sentinel2_green
# % required: yes
# % description: Sentinel-2 B3 (green) surface reflectance [0,1], resampled/aligned to alpha_map's region
# % guisection: Input
# %end

# %option G_OPT_R_INPUT
# % key: sentinel2_nir
# % required: no
# % description: Sentinel-2 B8 (NIR) surface reflectance [0,1] -- for standard NDWI (McFeeters 1996). Give either this or sentinel2_swir, not both
# % guisection: Input
# %end

# %option G_OPT_R_INPUT
# % key: sentinel2_swir
# % required: no
# % description: Sentinel-2 B11 (SWIR1) surface reflectance [0,1] -- for MNDWI (Xu 2006), more robust than NDWI for turbid/organic-laden POME ponds. Give either this or sentinel2_nir, not both
# % guisection: Input
# %end

# %option
# % key: water_threshold
# % type: double
# % required: no
# % answer: 0.0
# % description: NDWI/MNDWI threshold above which a pixel is classified as water/pond
# % guisection: Retrieval
# %end

# %option
# % key: percentile
# % type: double
# % required: no
# % answer: 90
# % description: Percentile of alpha_map defining the hotspot ("plume") pixel set to test against the water mask
# % guisection: Retrieval
# %end

# %option
# % key: artifact_fraction_threshold
# % type: double
# % required: no
# % answer: 0.8
# % description: If at least this fraction of hotspot pixels fall inside the water/pond mask, flag the hotspot as a likely retrieval artifact
# % guisection: Retrieval
# %end

# %option G_OPT_R_OUTPUT
# % key: water_mask_output
# % required: no
# % description: Optionally save the Sentinel-2-derived water/pond classification raster
# % guisection: Output
# %end

import grass.script as gs


def classify_water_sentinel2(green, nir=None, swir=None, threshold=0.0):
    """Boolean water/pond classification from Sentinel-2 reflectance:
    standard NDWI (McFeeters 1996: (green-nir)/(green+nir)) if `nir` is
    given, or MNDWI (Xu 2006: (green-swir)/(green+swir)) if `swir` is given
    instead -- MNDWI is generally more robust for turbid, organic-laden
    water bodies like POME ponds, which standard NDWI can under-detect.
    Returns the temporary index raster's name and the boolean mask's name.
    """
    if bool(nir) == bool(swir):
        gs.fatal("Give exactly one of sentinel2_nir (NDWI) or sentinel2_swir (MNDWI), not both/neither.")
    other = nir if nir else swir
    index_name = gs.tempname(8)
    gs.mapcalc(f"{index_name} = ({green} - {other}) / ({green} + {other})", overwrite=True, quiet=True)
    mask_name = gs.tempname(8)
    gs.mapcalc(f"{mask_name} = if({index_name} > {threshold}, 1, 0)", overwrite=True, quiet=True)
    return index_name, mask_name


def hotspot_water_overlap(alpha_map, water_mask, percentile):
    """Fraction of alpha_map's top-`percentile` ("hotspot") pixels that fall
    within the water_mask, plus the pixel-based Jaccard index (intersection
    over union) between the hotspot set and the water mask -- a high
    fraction/IoU means the methane enhancement's spatial footprint tracks
    the pond boundary, the artifact signature described in Valverde et al.
    2024 Fig. 4. alpha_map and water_mask must already share the same
    region/resolution (e.g. via r.resamp.stats onto alpha_map's grid)."""
    import numpy as np
    from grass.script import array as garray

    alpha = np.array(garray.array(mapname=alpha_map, dtype=np.float64, null=np.nan))
    water = np.array(garray.array(mapname=water_mask, dtype=np.float64, null=np.nan))
    if alpha.shape != water.shape:
        gs.fatal(
            f"<{alpha_map}> ({alpha.shape}) and <{water_mask}> ({water.shape}) do not "
            "share the same grid shape -- resample the water mask onto alpha_map's "
            "region/resolution first (e.g. r.resamp.stats)."
        )
    valid = np.isfinite(alpha) & np.isfinite(water)
    if not valid.any():
        gs.fatal("No overlapping valid pixels between alpha_map and water_mask.")
    threshold = np.nanpercentile(alpha[valid], percentile)
    # Strict ">": with a plateau of tied low values (e.g. a flat background),
    # ">=" against a percentile that lands on that plateau would include the
    # entire background, not just the genuine high tail. Real (continuous)
    # alpha data essentially never has exact ties, but fall back to ">=" if
    # ">" happens to select nothing (e.g. the percentile lands exactly on a
    # tied high plateau instead).
    hotspot = valid & (alpha > threshold)
    if not hotspot.any():
        hotspot = valid & (alpha >= threshold)
    water_bool = valid & (water > 0.5)

    n_hotspot = int(hotspot.sum())
    if n_hotspot == 0:
        gs.fatal(f"No pixels at or above the {percentile}th percentile of <{alpha_map}>.")
    n_in_water = int((hotspot & water_bool).sum())
    n_union = int((hotspot | water_bool).sum())
    fraction_in_water = n_in_water / n_hotspot
    iou = n_in_water / n_union if n_union else 0.0
    return fraction_in_water, iou, n_hotspot, n_in_water


def main():
    options, flags = gs.parser()

    alpha_map = options["alpha_map"]
    green = options["sentinel2_green"]
    nir = options["sentinel2_nir"] or None
    swir = options["sentinel2_swir"] or None
    threshold = float(options["water_threshold"])
    percentile = float(options["percentile"])
    artifact_fraction_threshold = float(options["artifact_fraction_threshold"])
    water_mask_output = options["water_mask_output"] or None

    index_name, mask_name = classify_water_sentinel2(green, nir=nir, swir=swir, threshold=threshold)
    try:
        fraction_in_water, iou, n_hotspot, n_in_water = hotspot_water_overlap(
            alpha_map, mask_name, percentile
        )
    finally:
        to_remove = [index_name] if water_mask_output else [index_name, mask_name]
        gs.run_command("g.remove", type="raster", name=",".join(to_remove), flags="f", quiet=True)
        if water_mask_output:
            gs.run_command("g.rename", raster=f"{mask_name},{water_mask_output}", quiet=True)

    likely_artifact = fraction_in_water >= artifact_fraction_threshold
    gs.message(
        f"Hotspot (top {100 - percentile:.0f}% of <{alpha_map}>): {n_hotspot} pixels, "
        f"{n_in_water} ({fraction_in_water * 100:.1f}%) inside the Sentinel-2 water/pond mask. "
        f"Pixel-based IoU with the water mask: {iou:.3f}.\n"
        + (
            f"LIKELY RETRIEVAL ARTIFACT: hotspot's spatial footprint tracks the pond "
            f"boundary (>= {artifact_fraction_threshold * 100:.0f}% inside water mask), "
            "consistent with the pond-albedo artifact pattern in Valverde et al. 2024 Fig. 4."
            if likely_artifact
            else
            f"NOT flagged as a pond-boundary artifact (< {artifact_fraction_threshold * 100:.0f}% "
            "inside water mask) -- the enhancement extends beyond what the pond surface alone "
            "would explain, consistent with (but not proof of) a genuine emission."
        )
    )


if __name__ == "__main__":
    main()
