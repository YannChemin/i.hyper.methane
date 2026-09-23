"""pytest tests for plume_artifact_check.py.

Run with: pytest testsuite/test_plume_artifact_check.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "plume_artifact_check.py"

grass = pytest.importorskip("grass.script")


@pytest.fixture
def grass_session(tmp_path):
    gisdb = tmp_path / "grassdata"
    gisdb.mkdir()
    project = "artifact_check_test"
    grass.core.create_project(str(gisdb), project, epsg="4326")
    session = grass.setup.init(gisdb / project / "PERMANENT")
    yield session
    session.finish()


def run_check(env: dict, overwrite: bool = False, **kwargs) -> subprocess.CompletedProcess:
    args = [sys.executable, str(MODULE_PATH)] + [f"{k}={v}" for k, v in kwargs.items()]
    if overwrite:
        args.append("--overwrite")
    return subprocess.run(args, env=env, capture_output=True, text=True)


def setup_rasters(env):
    grass.run_command("g.region", n=20, s=0, e=20, w=0, res=1, env=env)
    grass.mapcalc("green = 0.10", overwrite=True, env=env)
    grass.mapcalc(
        "nir = if(row() >= 8 && row() <= 12 && col() >= 8 && col() <= 12, 0.05, 0.35)",
        overwrite=True, env=env,
    )
    # Hotspot exactly matching the pond footprint -> artifact signature.
    grass.mapcalc(
        "alpha_artifact = if(row() >= 8 && row() <= 12 && col() >= 8 && col() <= 12, 0.9, 0.01)",
        overwrite=True, env=env,
    )
    # Hotspot extending well beyond the pond -> plausible real emission.
    grass.mapcalc(
        "alpha_real = if(row() >= 6 && row() <= 16 && col() >= 6 && col() <= 16, 0.9, 0.01)",
        overwrite=True, env=env,
    )


def test_flags_hotspot_matching_pond_boundary(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)
    setup_rasters(env)

    result = run_check(
        env, overwrite=True,
        alpha_map="alpha_artifact", sentinel2_green="green", sentinel2_nir="nir",
    )
    assert result.returncode == 0, result.stderr
    assert "LIKELY RETRIEVAL ARTIFACT" in result.stderr
    assert "100.0%" in result.stderr


def test_does_not_flag_hotspot_exceeding_pond_boundary(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)
    setup_rasters(env)

    result = run_check(
        env, overwrite=True,
        alpha_map="alpha_real", sentinel2_green="green", sentinel2_nir="nir",
    )
    assert result.returncode == 0, result.stderr
    assert "LIKELY RETRIEVAL ARTIFACT" not in result.stderr
    assert "NOT flagged" in result.stderr


def test_fails_loudly_with_both_nir_and_swir(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)
    setup_rasters(env)
    grass.mapcalc("swir = 0.2", overwrite=True, env=env)

    result = run_check(
        env, overwrite=True,
        alpha_map="alpha_artifact", sentinel2_green="green",
        sentinel2_nir="nir", sentinel2_swir="swir",
    )
    assert result.returncode != 0
    assert "exactly one of" in result.stderr
