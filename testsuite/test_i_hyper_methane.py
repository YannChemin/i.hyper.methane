"""pytest tests for i.hyper.methane.

Builds a small synthetic 3-band 3D raster (values chosen so the
band-depth result can be hand-verified) with an i.hyper.import-style
hyper.json wavelength sidecar, then runs the module against it in a
throwaway GRASS project.

Run with: pytest testsuite/test_i_hyper_methane.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "i.hyper.methane.py"

grass = pytest.importorskip("grass.script")


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


def test_band_depth_matches_hand_calculation(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)

    grass.run_command("g.region", n=10, s=0, e=10, w=0, res=1, t=3, b=0, tbres=1, flags="3", env=env)
    grass.mapcalc3d(
        "cube = if(z() < 1, 0.30, if(z() < 2, 0.15, 0.32))",
        overwrite=True, env=env,
    )

    gisenv = grass.gisenv(env=env)
    grid3_dir = Path(gisenv["GISDBASE"]) / gisenv["LOCATION_NAME"] / gisenv["MAPSET"] / "grid3" / "cube"
    grid3_dir.mkdir(parents=True, exist_ok=True)
    (grid3_dir / "hyper.json").write_text(json.dumps({"bands": {"wavelength": [2120, 2298, 2400]}}))

    result = run_module(env, input="cube", output="ch4_depth", overwrite=True)
    assert result.returncode == 0, result.stderr

    stats = grass.parse_command("r.univar", map="ch4_depth", flags="g", env=env)
    frac = (2298 - 2120) / (2400 - 2120)
    continuum = 0.30 + (0.32 - 0.30) * frac
    expected = 1.0 - (0.15 / continuum)
    assert float(stats["mean"]) == pytest.approx(expected, rel=1e-4)


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
        "cube = if(z() < 1, 0.30, if(z() < 2, 0.15, 0.32))",
        overwrite=True, env=env,
    )
    gisenv = grass.gisenv(env=env)
    grid3_dir = Path(gisenv["GISDBASE"]) / gisenv["LOCATION_NAME"] / gisenv["MAPSET"] / "grid3" / "cube"
    grid3_dir.mkdir(parents=True, exist_ok=True)
    (grid3_dir / "hyper.json").write_text(json.dumps({"bands": {"wavelength": [2120, 2298, 2400]}}))

    result = run_module(env, input="cube", output="ch4_bad", right_shoulder=3000, overwrite=True)
    assert result.returncode != 0
    assert "No band within" in result.stderr
