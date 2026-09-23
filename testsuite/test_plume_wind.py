"""pytest tests for plume_wind.py's orientation/consistency/flux math.

These deliberately avoid t.in.era5 (network + CDS/ARCO-ERA5 access, which
can take minutes) -- fetch_wind()/sample_wind_at_point() were validated
manually against real ERA5 data at the Colombia POM site from Valverde
et al. 2024 (8.614N, -73.680E, 2023-03-19: 0.41 m/s from 296.9 deg,
daily-mean -- broadly consistent with, though smoother than, the paper's
reported instantaneous 0.9 m/s from the southeast).

Run with: pytest testsuite/test_plume_wind.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "plume_wind.py"

grass = pytest.importorskip("grass.script")
np = pytest.importorskip("numpy")


@pytest.fixture
def grass_session(tmp_path):
    gisdb = tmp_path / "grassdata"
    gisdb.mkdir()
    project = "plume_wind_test"
    grass.core.create_project(str(gisdb), project, epsg="4326")
    session = grass.setup.init(gisdb / project / "PERMANENT")
    yield session
    session.finish()


@pytest.fixture
def mod(grass_session):
    spec = importlib.util.spec_from_file_location("plume_wind_impl", MODULE_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_orientation_detects_east_west_elongation(mod):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.uniform(-100, 100, n)
    y = rng.normal(0, 5, n)
    angle = mod.plume_orientation_deg(x, y)
    assert 80 <= angle <= 100  # ~90 deg from north = east-west


def test_orientation_detects_north_south_elongation(mod):
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(0, 5, n)
    y = rng.uniform(-100, 100, n)
    angle = mod.plume_orientation_deg(x, y)
    assert angle <= 10 or angle >= 170  # ~0/180 deg from north = north-south


def test_direction_resolves_sign_toward_ne(mod):
    # Source-relative offsets: a plume extending NE (dx>0, dy>0) from the
    # source at (0,0), decaying with distance, small floor elsewhere.
    rng = np.random.default_rng(2)
    n = 3000
    along = rng.uniform(0, 60, n)
    across = rng.normal(0, 4, n)
    ang = np.radians(45.0)
    dx = along * np.sin(ang) - across * np.cos(ang)
    dy = along * np.cos(ang) + across * np.sin(ang)
    values = np.exp(-along / 20.0) + 0.02

    direction = mod.plume_direction_deg(dx, dy, values)
    assert direction is not None
    assert 35 <= direction <= 55  # ~45 deg (NE), not the 225 deg unsigned-axis ambiguity


def test_direction_resolves_sign_toward_sw(mod):
    # Mirror image of the NE case: plume extends SW instead.
    rng = np.random.default_rng(3)
    n = 3000
    along = rng.uniform(0, 60, n)
    across = rng.normal(0, 4, n)
    ang = np.radians(45.0)
    dx = -(along * np.sin(ang) - across * np.cos(ang))
    dy = -(along * np.cos(ang) + across * np.sin(ang))
    values = np.exp(-along / 20.0) + 0.02

    direction = mod.plume_direction_deg(dx, dy, values)
    assert direction is not None
    assert 215 <= direction <= 235  # ~225 deg (SW)


def test_direction_none_for_symmetric_hotspot(mod):
    # A hotspot centered exactly on the source, symmetric in all directions,
    # has no resolvable sign -- plume_direction_deg() should say so (None)
    # rather than report an arbitrary axis.
    rng = np.random.default_rng(4)
    theta = rng.uniform(0, 2 * np.pi, 2000)
    r = rng.uniform(0, 20, 2000)
    dx = r * np.cos(theta)
    dy = r * np.sin(theta)
    values = np.ones(2000)
    assert mod.plume_direction_deg(dx, dy, values) is None


def test_direction_agreement_matches_and_mismatches(mod):
    diff, ok = mod.direction_agreement(45.0, 50.0, tolerance_deg=45.0)
    assert ok
    assert diff == pytest.approx(5.0)

    diff2, ok2 = mod.direction_agreement(45.0, 225.0, tolerance_deg=45.0)
    assert not ok2
    assert diff2 == pytest.approx(180.0)


def test_wind_consistency_aligned_and_perpendicular(mod):
    # East-west plume (90 deg), wind from the west (270) blows east -> aligned.
    diff, ok = mod.wind_consistency(90.0, 270.0, 45.0)
    assert ok
    assert diff < 10

    # Same plume, wind from the north (0) blows south -> perpendicular, not consistent.
    diff2, ok2 = mod.wind_consistency(90.0, 0.0, 45.0)
    assert not ok2
    assert diff2 > 80


def test_flux_none_without_calibration(mod):
    values = np.ones(50)
    assert mod.estimate_flux(values, 1.0, 100.0, 2.0, alpha_to_ppm_m=None) is None


def test_flux_positive_with_calibration(mod):
    values = np.ones(50)
    flux = mod.estimate_flux(values, pixel_area_m2=1.0, plume_area_m2=100.0, wind_speed_ms=2.0, alpha_to_ppm_m=1.0)
    assert flux is not None
    assert flux > 0


def test_ppm_m_to_kg_m2_positive_and_scales_linearly(mod):
    a = mod.ppm_m_to_kg_m2(1.0)
    b = mod.ppm_m_to_kg_m2(2.0)
    assert a > 0
    assert b == pytest.approx(2 * a, rel=1e-9)


def test_image_only_flag_skips_era5_and_reports_direction(grass_session):
    """CLI-level: -i must resolve a direction from the image and exit
    without touching ERA5 at all (no date required, no network needed)."""
    env = os.environ.copy()
    env.update(grass_session.env)

    # n/s bounded within +/-90 deg (this fixture's project is EPSG:4326).
    grass.run_command("g.region", n=50, s=-50, e=50, w=-50, res=1, env=env)
    rng = np.random.default_rng(5)
    n = 3000
    along = rng.uniform(0, 40, n)
    across = rng.normal(0, 4, n)
    ang = np.radians(45.0)
    dx = along * np.sin(ang) - across * np.cos(ang)
    dy = along * np.cos(ang) + across * np.sin(ang)
    val = np.exp(-along / 20.0) + 0.02
    cols = np.clip((dx).astype(int) + 50, 0, 99)
    rows = np.clip((-dy).astype(int) + 50, 0, 99)
    grid = np.zeros((100, 100))
    for r, c, v in zip(rows, cols, val):
        grid[r, c] = max(grid[r, c], v)

    import grass.script.array as garray
    a = garray.array(env=env)
    a[...] = grid
    a.write(mapname="alpha_ne_plume", overwrite=True)

    args = [
        sys.executable, str(MODULE_PATH), "-i",
        "alpha_map=alpha_ne_plume", "source=0,0", "radius=45", "percentile=70",
    ]
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "skipping ERA5" in result.stderr
    assert "Plume direction:" in result.stderr
    # ~45 deg NE, resolved (not the unsigned-axis fallback message)
    assert "resolved from the image" in result.stderr


def test_fails_loudly_without_date_or_image_only_flag(grass_session):
    env = os.environ.copy()
    env.update(grass_session.env)
    grass.run_command("g.region", n=10, s=0, e=10, w=0, res=1, env=env)
    grass.mapcalc("alpha_flat = 0.5", overwrite=True, env=env)

    args = [sys.executable, str(MODULE_PATH), "alpha_map=alpha_flat", "source=5,5"]
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "date=" in result.stderr
