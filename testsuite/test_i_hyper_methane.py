"""pytest tests for i.hyper.methane.

Builds small synthetic 3D rasters with an i.hyper.import-style hyper.json
wavelength sidecar, then runs the matched-filter module against them in a
throwaway GRASS project.

Run with: pytest testsuite/test_i_hyper_methane.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "i.hyper.methane.py"

grass = pytest.importorskip("grass.script")
np = pytest.importorskip("numpy")
garray = pytest.importorskip("grass.script.array")


@pytest.fixture
def grass_session(tmp_path):
    gisdb = tmp_path / "grassdata"
    gisdb.mkdir()
    project = "methane_test"
    grass.core.create_project(str(gisdb), project, epsg="4326")

    session = grass.setup.init(gisdb / project / "PERMANENT")
    yield session
    session.finish()


def run_module(env: dict, overwrite: bool = False, **kwargs) -> subprocess.CompletedProcess:
    args = [sys.executable, str(MODULE_PATH)] + [f"{k}={v}" for k, v in kwargs.items()]
    if overwrite:
        args.append("--overwrite")
    return subprocess.run(args, env=env, capture_output=True, text=True)


def write_hyper_json(env: dict, mapname: str, wavelengths_nm) -> None:
    gisenv = grass.gisenv(env=env)
    grid3_dir = Path(gisenv["GISDBASE"]) / gisenv["LOCATION_NAME"] / gisenv["MAPSET"] / "grid3" / mapname
    grid3_dir.mkdir(parents=True, exist_ok=True)
    (grid3_dir / "hyper.json").write_text(
        json.dumps({"bands": {"wavelength": [float(w) for w in wavelengths_nm]}})
    )


def test_matched_filter_flags_synthetic_plume(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)

    nbands, nrows, ncols = 9, 12, 12
    wavelengths = np.linspace(2120, 2400, nbands)

    grass.run_command(
        "g.region", n=nrows, s=0, e=ncols, w=0, res=1,
        t=nbands, b=0, tbres=1, flags="3", env=env,
    )

    rng = np.random.default_rng(42)
    mu = 100 + 5 * np.sin(np.linspace(0, 3, nbands))
    cube = mu[:, None, None] + rng.normal(scale=0.5, size=(nbands, nrows, ncols))

    # Inject a synthetic CH4 absorption dip, shaped like the module's default
    # Gaussian template (center 2298 nm, FWHM 40 nm), at a single pixel.
    sigma = 40 / 2.3548200450309493
    template = np.exp(-0.5 * ((wavelengths - 2298) / sigma) ** 2)
    template /= template.max()
    plume_row, plume_col = 6, 6
    cube[:, plume_row, plume_col] -= 8.0 * template * mu

    arr3d = garray.array3d(env=env)
    arr3d[...] = cube
    arr3d.write(mapname="cube", overwrite=True)
    write_hyper_json(env, "cube", wavelengths)

    result = run_module(env, input="cube", output="ch4_mf", overwrite=True)
    assert result.returncode == 0, result.stderr

    alpha = np.array(garray.array(mapname="ch4_mf", dtype=np.float64, null=np.nan, env=env))
    background = np.delete(alpha.flatten(), plume_row * ncols + plume_col)
    background = background[np.isfinite(background)]

    assert np.isfinite(alpha[plume_row, plume_col])
    assert alpha[plume_row, plume_col] > np.nanmean(background) + 4 * np.nanstd(background)


def test_fails_loudly_without_wavelength_metadata(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)

    grass.run_command("g.region", n=10, s=0, e=10, w=0, res=1, t=3, b=0, tbres=1, flags="3", env=env)
    grass.mapcalc3d("cube_nometa = z()", overwrite=True, env=env)

    result = run_module(env, input="cube_nometa", output="ch4_bad", overwrite=True)
    assert result.returncode != 0
    assert "No wavelength metadata found" in result.stderr


def test_fails_loudly_when_no_band_within_tolerance(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)

    grass.run_command("g.region", n=10, s=0, e=10, w=0, res=1, t=3, b=0, tbres=1, flags="3", env=env)
    grass.mapcalc3d(
        "cube = if(z() < 1, 100.0, if(z() < 2, 95.0, 102.0))",
        overwrite=True, env=env,
    )
    write_hyper_json(env, "cube", [2120, 2298, 2400])

    result = run_module(
        env, input="cube", output="ch4_bad",
        absorption_wavelength=1500, left_shoulder=1400, right_shoulder=1600,
        overwrite=True,
    )
    assert result.returncode != 0
    assert "No band within" in result.stderr


def test_fails_loudly_with_too_few_window_bands(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)

    grass.run_command("g.region", n=10, s=0, e=10, w=0, res=1, t=3, b=0, tbres=1, flags="3", env=env)
    grass.mapcalc3d(
        "cube = if(z() < 1, 100.0, if(z() < 2, 95.0, 102.0))",
        overwrite=True, env=env,
    )
    write_hyper_json(env, "cube", [2120, 2298, 2400])

    result = run_module(env, input="cube", output="ch4_bad", overwrite=True)
    assert result.returncode != 0
    assert "need at least" in result.stderr
