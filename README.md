## DESCRIPTION

*i.hyper.methane* computes a methane (CH4) absorption band-depth proxy
index from hyperspectral imagery imported as 3D raster maps (`raster_3d`)
by [i.hyper.import](i.hyper.import.html).

The module locates three bands in the input cube's SWIR range: two
continuum "shoulder" bands (*left_shoulder*, *right_shoulder*, default
2120 nm / 2400 nm) that bracket the CH4 absorption doublet, and one
absorption band (*absorption_wavelength*, default 2298 nm, the strongest
CH4 feature in that region). It linearly interpolates a continuum
reflectance between the two shoulders at the absorption wavelength, then
computes:

```
band_depth = 1 - (R_absorption / R_continuum)
```

Values near 0 indicate no CH4 absorption signature (continuum-level
reflectance); larger positive values indicate stronger absorption,
consistent with elevated methane along the sensor-to-surface path.

*i.hyper.methane* is part of the **i.hyper** module family. It was built
for flagging candidate methane hotspots (e.g. near oil/gas infrastructure)
in hyperspectral scenes, as a companion to the coarser-resolution,
column-retrieval-based Sentinel-5P methane mapping done outside GRASS by
the ALMA Energy toolset's `05_methane_hotspots` script.

## NOTES

**This module computes a relative band-depth proxy, not a calibrated
methane column retrieval.** A quantitative retrieval (e.g. in ppm-m)
requires a matched-filter against a unit absorption spectrum together
with a scene background covariance model -- the approach used by, for
example, EMIT's own CH4 plume product. That is a substantially larger
undertaking than this module attempts. Use *i.hyper.methane*'s output to
flag and rank candidate hotspot locations for further investigation, not
as an absolute methane concentration.

The module expects the input 3D raster to carry per-band wavelength
metadata, following the same convention as the rest of the *i.hyper*
family (an *i.hyper.import* `hyper.json` sidecar, or **wavelength**/
**unit** in each band's `r3.support` history). If no band lies within
*tolerance* nanometers of a requested wavelength, the module fails with
`G_fatal_error` rather than silently substituting a mismatched band --
this typically means the input's SWIR coverage does not reach the CH4
absorption region near 2100-2400 nm.

## EXAMPLE

Compute a methane band-depth index from a hyperspectral cube imported by
*i.hyper.import*, using the default absorption/shoulder wavelengths:

```sh
i.hyper.methane input=emit_scene output=emit_ch4_depth
```

Using a narrower continuum window and a stricter wavelength-matching
tolerance:

```sh
i.hyper.methane input=emit_scene output=emit_ch4_depth \
    left_shoulder=2150 right_shoulder=2350 tolerance=8
```

## SEE ALSO

[i.hyper.import](i.hyper.import.html),
[i.hyper.continuum](i.hyper.continuum.html),
[i.hyper.indices](i.hyper.indices.html),
[r.mapcalc](https://grass.osgeo.org/grass-stable/manuals/r.mapcalc.html),
[r3.support](https://grass.osgeo.org/grass-stable/manuals/r3.support.html)

## REFERENCES

- Thompson, D. R., et al. (2015). Real-time remote detection and
  measurement for airborne imaging spectroscopy: a case study with
  methane. *Atmospheric Measurement Techniques*, 8(10), 4383-4397.
- Thorpe, A. K., et al. (2013). Retrieval techniques for airborne
  imaging of methane concentrations using high spatial and moderate
  spectral resolution: application to AVIRIS. *Atmospheric Measurement
  Techniques*, 6(12), 3527-3546.

## AUTHORS

Created for the i.hyper module family, ALMA Energy project
