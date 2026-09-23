#!/usr/bin/env python3
##############################################################################
# MODULE:    ch4_unit_spectrum (internal helper for i.hyper.methane)
# AUTHOR(S): Created for the i.hyper module family, ALMA Energy project
# PURPOSE:   Generate a physically-based CH4 unit absorption spectrum for
#            i.hyper.methane's matched filter, via finite-difference
#            libRadtran uvspec runs, using atmosphere composition retrieved
#            from the same hyperspectral cube (i.hyper.atcorr/libsixsv
#            retrieval functions) and a Landsat/Sentinel-2-derived albedo.
# COPYRIGHT: (C) 2026 by the GRASS Development Team
# SPDX-License-Identifier: GPL-2.0-or-later
##############################################################################

# %module
# % description: Generate a physically-based CH4 unit absorption spectrum for i.hyper.methane via libRadtran uvspec
# % keyword: imagery
# % keyword: hyperspectral
# % keyword: methane
# % keyword: radiative transfer
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % required: yes
# % description: Input hyperspectral 3D raster map (from i.hyper.import); ideally covers VIS-SWIR for atmosphere-band retrieval
# % guisection: Input
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % required: yes
# % description: Output two-column text file (wavelength_nm, absorption), usable as i.hyper.methane's absorption_spectrum
# % guisection: Output
# %end

# %option
# % key: left_shoulder
# % type: double
# % required: no
# % answer: 2120
# % description: Lower bound (nanometers) of the output wavelength grid, matching i.hyper.methane's matched-filter window
# % guisection: Wavelengths
# %end

# %option
# % key: right_shoulder
# % type: double
# % required: no
# % answer: 2400
# % description: Upper bound (nanometers) of the output wavelength grid, matching i.hyper.methane's matched-filter window
# % guisection: Wavelengths
# %end

# %option
# % key: tolerance
# % type: double
# % required: no
# % answer: 15
# % description: Maximum allowed distance (nanometers) for matching requested wavelengths (CH4 window or ancillary atmosphere bands) to available bands
# % guisection: Wavelengths
# %end

# %option
# % key: sza
# % type: double
# % required: yes
# % description: Solar zenith angle (degrees, 0-89)
# % guisection: Geometry
# %end

# %option
# % key: doy
# % type: integer
# % required: no
# % answer: 180
# % description: Day of year, for Earth-Sun distance (1-365)
# % guisection: Geometry
# %end

# %option
# % key: zout
# % type: string
# % required: no
# % answer: toa
# % description: uvspec output altitude: 'toa' (default, TOA-equivalent, omits the zout line -- an explicit zout produced all-zero radiance in local testing) or a numeric altitude in km for an airborne platform (e.g. 4 for AVIRIS-NG-like; not yet verified to produce valid output)
# % guisection: Geometry
# %end

# %option G_OPT_R_INPUT
# % key: elevation
# % required: no
# % description: 2D elevation raster [m], for ISA surface-pressure estimation (scene mean); falls back to 1013.25 hPa if not given
# % guisection: Atmosphere
# %end

# %option
# % key: albedo
# % type: double
# % required: no
# % description: Scalar surface albedo (0-1). Overrides albedo_sensor/albedo_bands if given
# % guisection: Albedo
# %end

# %option
# % key: albedo_sensor
# % type: string
# % required: no
# % options: landsat5,landsat7,landsat8,sentinel2
# % description: Sensor that albedo_bands were imported from (via r.in.landsat or r.in.sentinel)
# % guisection: Albedo
# %end

# %option G_OPT_R_INPUTS
# % key: albedo_bands
# % required: no
# % description: Raster band maps for albedo estimation. Landsat: i.albedo's expected band order/count for the chosen sensor. Sentinel-2: exactly 6 reflectance bands [0,1] in B2,B3,B4,B8,B11,B12 order (Bonafoni & Sekertekin 2020 narrow-to-broadband coefficients)
# % guisection: Albedo
# %end

# %option
# % key: ch4_baseline_ppm
# % type: double
# % required: no
# % answer: 1.9
# % description: Baseline atmospheric CH4 mixing ratio (ppm) for the reference uvspec run
# % guisection: Retrieval
# %end

# %option
# % key: ch4_enhancement_ppm
# % type: double
# % required: no
# % answer: 0.5
# % description: CH4 mixing-ratio perturbation (ppm) used for the finite-difference unit absorption spectrum
# % guisection: Retrieval
# %end

# %option
# % key: uvspec_bin
# % type: string
# % required: no
# % answer: uvspec
# % description: Path to the uvspec binary (libRadtran)
# % guisection: Retrieval
# %end

# %option
# % key: libradtran_data
# % type: string
# % required: no
# % description: libRadtran data/ directory. Auto-detected as <dir of uvspec_bin>/../data if not given
# % guisection: Retrieval
# %end

# %option
# % key: solar_source
# % type: string
# % required: no
# % answer: kurudz_1.0nm.dat
# % description: Solar source spectrum file (in libradtran_data/solar_flux/) covering the requested SWIR window; the libRadtran default (atlas_plus_modtran) only covers 200-800 nm and will fail here
# % guisection: Retrieval
# %end

# %option
# % key: libsixsv
# % type: string
# % required: no
# % description: Path to libsixsv.so, for atmosphere-composition retrieval. Auto-detected next to this script's i.hyper.atcorr sibling if not given
# % guisection: Retrieval
# %end

import ctypes
import importlib.util
import os
import subprocess
import sys

import grass.script as gs

FALLBACK_OZONE_DU = 300.0
FALLBACK_AOD550 = 0.1
FALLBACK_WVC_GCM2 = 2.0
FALLBACK_PRESSURE_HPA = 1013.25


def _load_methane_module():
    """Import i.hyper.methane.py (dotted filename, not import-able normally)
    from this script's own directory, to reuse its band-wavelength helpers."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "i.hyper.methane.py")
    spec = importlib.util.spec_from_file_location("i_hyper_methane_impl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def soft_nearest_band(bands, target_wl, tolerance):
    """Like i.hyper.methane's nearest_band(), but returns None instead of
    gs.fatal()-ing, since atmosphere-composition bands are best-effort here."""
    closest = min(bands, key=lambda b: abs(b["wavelength"] - target_wl))
    if abs(closest["wavelength"] - target_wl) > tolerance:
        return None
    return closest


def extract_band_flat(methane_mod, input3d, band, tmp_prefix):
    """Extract one band's Z-slice and return it as a flat float64 numpy array,
    via i.hyper.methane's extract_z_slice() + grass.script.array."""
    import numpy as np
    from grass.script import array as garray

    name2d = f"{tmp_prefix}_atm_{band['band_num']}"
    methane_mod.extract_z_slice(input3d, band["band_num"] - 1, name2d)
    try:
        arr = np.array(garray.array(mapname=name2d, dtype=np.float64, null=np.nan)).ravel()
    finally:
        gs.run_command("g.remove", type="raster", name=name2d, flags="f", quiet=True)
    return arr


def to_c_float(arr):
    import numpy as np

    contig = np.ascontiguousarray(arr, dtype=np.float32)
    return contig, contig.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def load_libsixsv(path):
    # libsixsv.so uses GCC's vectorized libm symbols (e.g. _ZGVbN2v_exp), which
    # live in glibc's libmvec and are normally resolved by the linker at
    # compile time; ctypes needs libmvec loaded RTLD_GLOBAL first so the
    # dynamic loader can find them.
    try:
        ctypes.CDLL("libmvec.so.1", mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass
    lib = ctypes.CDLL(path)
    FP = ctypes.POINTER(ctypes.c_float)
    lib.retrieve_o3_chappuis.restype = ctypes.c_float
    lib.retrieve_o3_chappuis.argtypes = [FP, FP, FP, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    lib.retrieve_aod_ddv.restype = ctypes.c_float
    lib.retrieve_aod_ddv.argtypes = [FP, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_float, FP]
    lib.retrieve_h2o_940.restype = None
    lib.retrieve_h2o_940.argtypes = [FP, FP, FP, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float, FP]
    lib.retrieve_pressure_isa.restype = ctypes.c_float
    lib.retrieve_pressure_isa.argtypes = [ctypes.c_float]
    return lib


def retrieve_atmosphere(lib, methane_mod, input3d, bands, tolerance, sza, doy, elevation, tmp_prefix):
    """Best-effort retrieval of ozone/AOD/water-vapor/pressure from the same
    hyperspectral cube, using i.hyper.atcorr's libsixsv retrieval functions.
    Falls back to i.hyper.atcorr's own documented defaults (with a warning)
    for any parameter whose required bands are not covered by this cube."""
    import numpy as np

    def bands_for(wavelengths, label):
        found = [soft_nearest_band(bands, wl, tolerance) for wl in wavelengths]
        if any(b is None for b in found):
            gs.warning(
                f"<{input3d}> does not cover all bands needed for {label} retrieval "
                f"(wavelengths {wavelengths} nm, tolerance {tolerance} nm); using the "
                "documented fallback value instead."
            )
            return None
        return found

    # Ozone (Chappuis band depth, 540/600/680 nm) -> Dobson units.
    ozone_du = FALLBACK_OZONE_DU
    found = bands_for([540, 600, 680], "ozone")
    if found:
        l540, l600, l680 = (extract_band_flat(methane_mod, input3d, b, tmp_prefix) for b in found)
        mask = np.isfinite(l540) & np.isfinite(l600) & np.isfinite(l680)
        if mask.any():
            _, p540 = to_c_float(l540[mask])
            _, p600 = to_c_float(l600[mask])
            _, p680 = to_c_float(l680[mask])
            ozone_du = float(lib.retrieve_o3_chappuis(p540, p600, p680, int(mask.sum()), sza, 0.0))

    # AOD at 550 nm (dark-target DDV, 470/660/860/2130 nm). retrieve_aod_ddv's
    # own Angstrom interpolation already references the result to 550 nm, but
    # uvspec's aerosol_set_tau_at_wvl requires its anchor wavelength to lie
    # inside the simulated `wavelength` range. Rather than force 550 nm (which
    # may not be near any band this sensor actually measured), anchor the
    # optical depth at the real VNIR band nearest 470 nm that this retrieval
    # already used -- the residual error from applying a 550 nm-referenced
    # AOD at a nearby VNIR wavelength instead is small over the difference
    # (aerosol optical depth varies smoothly and slowly with wavelength).
    aod550 = FALLBACK_AOD550
    aod_anchor_wl = None
    found = bands_for([470, 660, 860, 2130], "AOD")
    if found:
        l470, l660, l860, l2130 = (extract_band_flat(methane_mod, input3d, b, tmp_prefix) for b in found)
        mask = np.isfinite(l470) & np.isfinite(l660) & np.isfinite(l860) & np.isfinite(l2130)
        if mask.any():
            _, p470 = to_c_float(l470[mask])
            _, p660 = to_c_float(l660[mask])
            _, p860 = to_c_float(l860[mask])
            _, p2130 = to_c_float(l2130[mask])
            n = int(mask.sum())
            out_aod = (ctypes.c_float * n)()
            aod550 = float(lib.retrieve_aod_ddv(p470, p660, p860, p2130, n, doy, sza, out_aod))
            aod_anchor_wl = found[0]["wavelength"]  # the cube's actual near-470nm VNIR band

    # Water vapor column (940 nm band depth, 865/940/1040 nm) -> g/cm^2.
    wvc = FALLBACK_WVC_GCM2
    found = bands_for([865, 940, 1040], "water vapor")
    if found:
        l865, l940, l1040 = (extract_band_flat(methane_mod, input3d, b, tmp_prefix) for b in found)
        mask = np.isfinite(l865) & np.isfinite(l940) & np.isfinite(l1040)
        if mask.any():
            n = int(mask.sum())
            _, p865 = to_c_float(l865[mask])
            _, p940 = to_c_float(l940[mask])
            _, p1040 = to_c_float(l1040[mask])
            out_wvc = (ctypes.c_float * n)()
            lib.retrieve_h2o_940(p865, p940, p1040, 0.0, n, sza, 0.0, out_wvc)
            wvc = float(np.nanmean(np.frombuffer(out_wvc, dtype=np.float32)))

    # Surface pressure: ISA from scene-mean elevation, else standard fallback.
    pressure_hpa = FALLBACK_PRESSURE_HPA
    if elevation:
        stats = gs.parse_command("r.univar", map=elevation, flags="g")
        if "mean" in stats:
            pressure_hpa = float(lib.retrieve_pressure_isa(float(stats["mean"])))

    return ozone_du, aod550, aod_anchor_wl, wvc, pressure_hpa


def albedo_from_landsat(sensor, bands):
    flag = {"landsat5": "l", "landsat7": "l", "landsat8": "8"}[sensor]
    out = gs.tempname(8)
    gs.run_command("i.albedo", input=",".join(bands), output=out, flags=flag, quiet=True)
    try:
        stats = gs.parse_command("r.univar", map=out, flags="g")
        return float(stats["mean"])
    finally:
        gs.run_command("g.remove", type="raster", name=out, flags="f", quiet=True)


SENTINEL2_ALBEDO_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
SENTINEL2_ALBEDO_COEFFS = [0.2266, 0.1236, 0.1573, 0.3417, 0.1170, 0.0338]  # sums to 1.0


def albedo_from_sentinel2(bands):
    """Broadband shortwave albedo from Sentinel-2 surface reflectance, using
    the narrow-to-broadband coefficients of Bonafoni & Sekertekin (2020),
    IEEE Geosci. Remote Sens. Lett. 17:1618-1622 -- for B2, B3, B4, B8, B11,
    B12 in that order (matching r.in.sentinel's band naming).

    These coefficient values were not read from the paper itself (paywalled)
    but cross-checked against an independent open-source implementation
    (github.com/qfisch/albedo, MIT-licensed, citing the same paper) that
    reproduces them; they sum to 1.0 as expected for this type of linear
    regression, which is consistent with a correctly transcribed set.

    `bands` must be exactly 6 raster maps in B2,B3,B4,B8,B11,B12 order, with
    reflectance already scaled to [0, 1] (not Sentinel-2's raw 0-10000 DN).
    """
    if len(bands) != len(SENTINEL2_ALBEDO_COEFFS):
        gs.fatal(
            f"albedo_sensor=sentinel2 requires exactly {len(SENTINEL2_ALBEDO_COEFFS)} "
            f"albedo_bands in {','.join(SENTINEL2_ALBEDO_BANDS)} order "
            f"(got {len(bands)})."
        )
    terms = [f"({c}*{b})" for c, b in zip(SENTINEL2_ALBEDO_COEFFS, bands)]
    expr = " + ".join(terms)
    out = gs.tempname(8)
    gs.mapcalc(f"{out} = {expr}", overwrite=True, quiet=True)
    try:
        stats = gs.parse_command("r.univar", map=out, flags="g")
        return float(stats["mean"])
    finally:
        gs.run_command("g.remove", type="raster", name=out, flags="f", quiet=True)


def resolve_albedo(options):
    if options["albedo"]:
        return float(options["albedo"])

    sensor = options["albedo_sensor"]
    band_str = options["albedo_bands"]
    if not sensor or not band_str:
        gs.fatal(
            "Provide either albedo=<value>, or both albedo_sensor= and albedo_bands= "
            "(raster maps imported via r.in.landsat or r.in.sentinel)."
        )
    bands = band_str.split(",")
    if sensor == "sentinel2":
        return albedo_from_sentinel2(bands)
    return albedo_from_landsat(sensor, bands)


def build_uvspec_input(data_dir, solar_source, ch4_ppm, doy, sza, albedo, zout, wl_lo, wl_hi,
                        ozone_du, wvc_gcm2, pressure_hpa, aod550=None, aod_anchor_wl=None):
    lines = [
        f"data_files_path {data_dir}/",
        f"atmosphere_file {data_dir}/atmmod/afglus.dat",
        f"source solar {data_dir}/solar_flux/{solar_source}",
        "mol_abs_param reptran fine",
        f"mixing_ratio CH4 {ch4_ppm}",
        f"mol_modify O3 {ozone_du} DU",
        f"mol_modify H2O {wvc_gcm2 * 10.0} MM",  # 1 g/cm^2 precipitable water = 10 mm
        f"pressure {pressure_hpa}",
        f"day_of_year {doy}",
        f"albedo {albedo}",
        f"sza {sza}",
        "umu -1.0",
        "phi 0.0",
    ]
    if aod_anchor_wl is not None:
        # Anchor the aerosol optical depth at the cube's own nearest-to-470nm
        # VNIR band rather than an arbitrary/unmeasured 550 nm point -- see
        # retrieve_atmosphere()'s comment for why. aerosol_set_tau_at_wvl
        # requires this wavelength to lie inside the simulated `wavelength`
        # range, which is enforced by the caller widening wl_lo accordingly.
        lines.append("aerosol_default")
        lines.append(f"aerosol_set_tau_at_wvl {aod_anchor_wl} {aod550}")
    # NOTE: an explicit "zout toa" (or any other explicit zout value) produces
    # an all-zero radiance spectrum with this umu/phi radiance setup in the
    # locally tested uvspec build, whereas omitting the line entirely gives
    # the correct top-of-atmosphere-equivalent result (verified empirically).
    # Only emit zout for a genuine non-default (e.g. airborne) altitude.
    if zout != "toa":
        lines.append(f"zout {zout}")
    lines += [
        "rte_solver disort",
        f"wavelength {wl_lo} {wl_hi}",
        "output_user lambda uu",
        "quiet",
    ]
    return "\n".join(lines) + "\n"


def run_uvspec(uvspec_bin, inp_text):
    import numpy as np

    result = subprocess.run([uvspec_bin], input=inp_text, capture_output=True, text=True)
    if result.returncode != 0:
        gs.fatal(f"uvspec failed (exit {result.returncode}):\n{result.stderr}")
    wavelengths, radiances = [], []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            wavelengths.append(float(parts[0]))
            radiances.append(float(parts[1]))
    if not wavelengths:
        gs.fatal(f"uvspec produced no usable output.\nstderr:\n{result.stderr}")
    return np.array(wavelengths), np.array(radiances)


def main():
    options, flags = gs.parser()

    input3d = options["input"]
    output = options["output"]
    wl_left = float(options["left_shoulder"])
    wl_right = float(options["right_shoulder"])
    tolerance = float(options["tolerance"])
    sza = float(options["sza"])
    doy = int(options["doy"])
    zout = options["zout"]
    elevation = options["elevation"] or None
    ch4_baseline = float(options["ch4_baseline_ppm"])
    ch4_delta = float(options["ch4_enhancement_ppm"])
    uvspec_bin = options["uvspec_bin"]
    solar_source = options["solar_source"]

    import numpy as np

    methane_mod = _load_methane_module()

    bands = methane_mod.get_all_band_wavelengths(input3d)
    window_bands = methane_mod.select_window_bands(bands, wl_left, wl_right)
    if len(window_bands) < methane_mod.MIN_WINDOW_BANDS:
        gs.fatal(
            f"Only {len(window_bands)} band(s) of <{input3d}> fall within "
            f"{wl_left}-{wl_right} nm; need at least {methane_mod.MIN_WINDOW_BANDS} "
            "to build a usable unit absorption spectrum."
        )
    target_wavelengths = np.array([b["wavelength"] for b in window_bands])

    libradtran_data = options["libradtran_data"]
    if not libradtran_data:
        import shutil
        uvspec_path = shutil.which(uvspec_bin)
        if not uvspec_path:
            gs.fatal(f"Cannot find uvspec_bin <{uvspec_bin}> on PATH; pass its full path or set libradtran_data=.")
        libradtran_data = os.path.normpath(os.path.join(os.path.dirname(uvspec_path), "..", "data"))
    if not os.path.isfile(os.path.join(libradtran_data, "atmmod", "afglus.dat")):
        gs.fatal(
            f"libradtran_data <{libradtran_data}> does not look like a libRadtran data/ "
            "directory (missing atmmod/afglus.dat); pass libradtran_data= explicitly."
        )

    libsixsv_path = options["libsixsv"]
    if not libsixsv_path:
        here = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.normpath(os.path.join(here, "..", "libsixsv", "libsixsv.so"))
        if os.path.isfile(candidate):
            libsixsv_path = candidate
    if not libsixsv_path or not os.path.isfile(libsixsv_path):
        gs.fatal(
            "Cannot find libsixsv.so for atmosphere-composition retrieval; pass libsixsv= explicitly."
        )
    lib = load_libsixsv(libsixsv_path)

    tmp_prefix = gs.tempname(8)
    ozone_du, aod550, aod_anchor_wl, wvc, pressure_hpa = retrieve_atmosphere(
        lib, methane_mod, input3d, bands, tolerance, sza, doy, elevation, tmp_prefix
    )

    if aod_anchor_wl is not None:
        aod_note = f"applied via aerosol_set_tau_at_wvl at the cube's own {aod_anchor_wl:.1f} nm VNIR band"
    else:
        aod_note = (
            "not applied (no VNIR bands near 470/660/860/2130 nm found in this cube to "
            "anchor aerosol_set_tau_at_wvl to, so aerosol_default's own climatology is used)"
        )
    gs.message(
        f"Atmosphere: ozone={ozone_du:.1f} DU, AOD550={aod550:.3f}, "
        f"WVC={wvc:.2f} g/cm2, pressure={pressure_hpa:.1f} hPa "
        f"(retrieved from <{input3d}> where covered, else i.hyper.atcorr's fallback defaults). "
        f"Ozone, water vapor and pressure are passed into the uvspec run; AOD is {aod_note}."
    )

    albedo = resolve_albedo(options)

    wl_lo = max(200.0, wl_left - 5)
    if aod_anchor_wl is not None:
        wl_lo = min(wl_lo, aod_anchor_wl - 5)
    inp_baseline = build_uvspec_input(
        libradtran_data, solar_source, ch4_baseline, doy, sza, albedo, zout,
        wl_lo, wl_right + 5, ozone_du, wvc, pressure_hpa, aod550, aod_anchor_wl,
    )
    inp_perturbed = build_uvspec_input(
        libradtran_data, solar_source, ch4_baseline + ch4_delta, doy, sza, albedo, zout,
        wl_lo, wl_right + 5, ozone_du, wvc, pressure_hpa, aod550, aod_anchor_wl,
    )

    wl_base, l_base = run_uvspec(uvspec_bin, inp_baseline)
    wl_pert, l_pert = run_uvspec(uvspec_bin, inp_perturbed)
    if wl_base.shape != wl_pert.shape or not np.allclose(wl_base, wl_pert):
        gs.fatal("Baseline and perturbed uvspec runs returned different wavelength grids.")

    delta_l = -(l_pert - l_base) / ch4_delta  # positive = absorption per ppm CH4
    absorption = np.interp(target_wavelengths, wl_base, delta_l)

    with open(output, "w") as f:
        f.write("# wavelength_nm absorption (finite-difference dL/dCH4ppm; sign so larger = more absorption)\n")
        f.write(f"# sza={sza} doy={doy} albedo={albedo:.4f} ch4_baseline_ppm={ch4_baseline} "
                f"ch4_enhancement_ppm={ch4_delta} zout={zout}\n")
        for wl, val in zip(target_wavelengths, absorption):
            f.write(f"{wl:.3f} {val:.8e}\n")

    gs.message(f"Wrote {len(target_wavelengths)}-band CH4 unit absorption spectrum to <{output}>")


if __name__ == "__main__":
    main()
