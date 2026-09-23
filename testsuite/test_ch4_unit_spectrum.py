"""pytest tests for ch4_unit_spectrum.py (i.hyper.methane's uvspec-based
CH4 unit absorption spectrum generator).

These tests drive the real uvspec binary and libsixsv.so, so they are
skipped outright when those external dependencies are not present at
their default locations (this is heavy machinery -- a libRadtran install
with fine-resolution reptran CH4 data, plus a built libsixsv.so -- not
expected in every environment that might check out this addon).

Run with: pytest testsuite/test_ch4_unit_spectrum.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = MODULE_DIR / "ch4_unit_spectrum.py"
METHANE_PATH = MODULE_DIR / "i.hyper.methane.py"

_DEV_UVSPEC = Path.home() / "dev/libRadtran-2.0.6/bin/uvspec"
UVSPEC_BIN = os.environ.get("UVSPEC_BIN") or (
    str(_DEV_UVSPEC) if _DEV_UVSPEC.is_file() else (shutil.which("uvspec") or str(_DEV_UVSPEC))
)
LIBRADTRAN_SOLAR_FLUX = Path(UVSPEC_BIN).resolve().parent.parent / "data" / "solar_flux" / "kurudz_1.0nm.dat"
LIBSIXSV_SO = Path.home() / "dev/libsixsv/libsixsv.so"

grass = pytest.importorskip("grass.script")
np = pytest.importorskip("numpy")
garray = pytest.importorskip("grass.script.array")

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(UVSPEC_BIN) and LIBRADTRAN_SOLAR_FLUX.is_file() and LIBSIXSV_SO.is_file()),
    reason="uvspec / libRadtran reptran-fine data / libsixsv.so not found at expected local paths",
)


@pytest.fixture
def grass_session(tmp_path):
    gisdb = tmp_path / "grassdata"
    gisdb.mkdir()
    project = "ch4_unit_spectrum_test"
    grass.core.create_project(str(gisdb), project, epsg="4326")

    session = grass.setup.init(gisdb / project / "PERMANENT")
    yield session
    session.finish()


def run_generator(env: dict, overwrite: bool = False, **kwargs) -> subprocess.CompletedProcess:
    args = [sys.executable, str(MODULE_PATH)] + [f"{k}={v}" for k, v in kwargs.items()]
    if overwrite:
        args.append("--overwrite")
    return subprocess.run(args, env=env, capture_output=True, text=True)


def run_methane(env: dict, overwrite: bool = False, **kwargs) -> subprocess.CompletedProcess:
    args = [sys.executable, str(METHANE_PATH)] + [f"{k}={v}" for k, v in kwargs.items()]
    if overwrite:
        args.append("--overwrite")
    return subprocess.run(args, env=env, capture_output=True, text=True)


def write_swir_cube(env: dict, mapname: str, nbands=9, nrows=4, ncols=4):
    wavelengths = np.linspace(2120, 2400, nbands)
    grass.run_command(
        "g.region", n=nrows, s=0, e=ncols, w=0, res=1,
        t=nbands, b=0, tbres=1, flags="3", env=env,
    )
    cube = garray.array3d(env=env)
    cube[...] = 100.0
    cube.write(mapname=mapname, overwrite=True)

    gisenv = grass.gisenv(env=env)
    grid3_dir = Path(gisenv["GISDBASE"]) / gisenv["LOCATION_NAME"] / gisenv["MAPSET"] / "grid3" / mapname
    grid3_dir.mkdir(parents=True, exist_ok=True)
    (grid3_dir / "hyper.json").write_text(
        json.dumps({"bands": {"wavelength": wavelengths.tolist()}})
    )
    return wavelengths


def test_generates_physically_sensible_ch4_spectrum(grass_session, tmp_path):
    env = os.environ.copy()
    env.update(grass_session.env)

    write_swir_cube(env, "swir_cube")
    out_file = tmp_path / "ch4_unit_absorption.txt"

    result = run_generator(
        env, overwrite=True,
        input="swir_cube", output=str(out_file),
        sza=30, doy=78, albedo=0.15, uvspec_bin=UVSPEC_BIN,
    )
    assert result.returncode == 0, result.stderr
    assert out_file.is_file()

    rows = [line.split() for line in out_file.read_text().splitlines() if not line.startswith("#")]
    wavelengths = np.array([float(r[0]) for r in rows])
    absorption = np.array([float(r[1]) for r in rows])

    assert len(rows) == 9
    assert np.all(np.isfinite(absorption))
    # Real CH4 absorption in this window is not uniformly zero, and should
    # be concentrated near the 2298 nm doublet rather than at the shoulders.
    assert np.any(absorption > 0)
    peak_wl = wavelengths[np.argmax(absorption)]
    assert 2250 <= peak_wl <= 2350


def test_aod_anchored_to_real_vnir_band(grass_session, tmp_path):
    """AOD retrieved from the cube's own VNIR bands should be applied via
    aerosol_set_tau_at_wvl at one of those real bands (not an arbitrary,
    possibly-unmeasured 550 nm point), when the cube covers VNIR."""
    env = os.environ.copy()
    env.update(grass_session.env)

    swir = np.linspace(2120, 2400, 9)
    vnir = [470, 540, 600, 660, 680, 865, 940, 1040, 2130]
    wavelengths = np.array(sorted(vnir + list(swir)))
    nbands = len(wavelengths)
    grass.run_command(
        "g.region", n=4, s=0, e=4, w=0, res=1,
        t=nbands, b=0, tbres=1, flags="3", env=env,
    )
    cube = garray.array3d(env=env)
    cube[...] = 0.1
    cube.write(mapname="full_cube", overwrite=True)
    gisenv = grass.gisenv(env=env)
    grid3_dir = Path(gisenv["GISDBASE"]) / gisenv["LOCATION_NAME"] / gisenv["MAPSET"] / "grid3" / "full_cube"
    grid3_dir.mkdir(parents=True, exist_ok=True)
    (grid3_dir / "hyper.json").write_text(json.dumps({"bands": {"wavelength": wavelengths.tolist()}}))

    out_file = tmp_path / "ch4_unit_absorption.txt"
    result = run_generator(
        env, overwrite=True,
        input="full_cube", output=str(out_file),
        sza=30, doy=78, albedo=0.15, uvspec_bin=UVSPEC_BIN,
    )
    assert result.returncode == 0, result.stderr
    assert "AOD is applied via aerosol_set_tau_at_wvl" in result.stderr
    assert "470.0 nm VNIR band" in result.stderr
    assert out_file.is_file()


def test_fails_loudly_without_albedo(grass_session, tmp_path):
    env = os.environ.copy()
    env.update(grass_session.env)

    write_swir_cube(env, "swir_cube")
    out_file = tmp_path / "ch4_unit_absorption.txt"

    result = run_generator(
        env, overwrite=True,
        input="swir_cube", output=str(out_file),
        sza=30, doy=78, uvspec_bin=UVSPEC_BIN,
    )
    assert result.returncode != 0
    assert "albedo" in result.stderr.lower()
    assert not out_file.exists()


def test_generated_spectrum_feeds_i_hyper_methane(grass_session, tmp_path):
    env = os.environ.copy()
    env.update(grass_session.env)

    nbands, nrows, ncols = 9, 12, 12
    wavelengths = write_swir_cube(env, "cube", nbands=nbands, nrows=nrows, ncols=ncols)

    rng = np.random.default_rng(1)
    mu = 100 + 5 * np.sin(np.linspace(0, 3, nbands))
    cube = garray.array3d(env=env)
    cube[...] = mu[:, None, None] + rng.normal(scale=0.5, size=(nbands, nrows, ncols))
    cube.write(mapname="cube", overwrite=True)

    out_file = tmp_path / "ch4_unit_absorption.txt"
    gen = run_generator(
        env, overwrite=True,
        input="cube", output=str(out_file),
        sza=30, doy=78, albedo=0.15, uvspec_bin=UVSPEC_BIN,
    )
    assert gen.returncode == 0, gen.stderr

    mf = run_methane(
        env, overwrite=True,
        input="cube", output="ch4_mf", absorption_spectrum=str(out_file),
    )
    assert mf.returncode == 0, mf.stderr

    stats = grass.parse_command("r.univar", map="ch4_mf", flags="g", env=env)
    assert int(stats["cells"]) == nrows * ncols
    assert int(stats.get("null_cells", 0)) == 0
