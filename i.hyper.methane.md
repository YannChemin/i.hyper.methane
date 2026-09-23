## DESCRIPTION

*i.hyper.methane* computes a methane (CH4) matched-filter enhancement map
from hyperspectral imagery imported as 3D raster maps (`raster_3d`) by
[i.hyper.import](i.hyper.import.html).

The module selects all bands of the input cube whose wavelength falls
within a SWIR spectral window (*left_shoulder* to *right_shoulder*,
default 2120-2400 nm, bracketing the CH4 absorption doublet), then applies
a background-covariance matched filter (Thorpe et al. 2013, Thompson et
al. 2015) across that window:

```
alpha(x) = (x - mu)^T C^-1 t / (t^T C^-1 t)
```

where, for each pixel's radiance vector *x* over the selected bands:

- *mu* is the scene background mean radiance spectrum,
- *C* is the background covariance matrix of (x - mu) across the scene,
  regularized with a diagonal loading term controlled by *shrinkage*,
- *t = -mu \* k* is the target signature, the linearized (optically-thin)
  radiance perturbation expected per unit methane path-concentration
  enhancement, built from a unit absorption-strength template *k*
  (peak-normalized to 1).

By default, *k* is a synthetic Gaussian centered at
*absorption_wavelength* (default 2298 nm, the strongest CH4 feature in
this region) with full-width-at-half-maximum *template_fwhm* (default
40 nm) -- a simplified stand-in for a real gas cross-section curve. For a
more physically faithful retrieval, supply *absorption_spectrum*: a
two-column text file (wavelength_nm, absorption) with a real CH4 unit
absorption spectrum (e.g. derived from HITRAN, or the "unit spectrum of
methane absorption" used in matched-filter literature); the module
interpolates it onto the input's own band wavelengths.

*alpha* is a relative enhancement statistic: values near 0 indicate no
detected CH4 signature; larger positive values indicate stronger
correlation with the absorption template, consistent with elevated
methane along the sensor-to-surface path.

*i.hyper.methane* is part of the **i.hyper** module family. It was built
for flagging candidate methane hotspots (e.g. near oil/gas infrastructure
or palm oil mill effluent ponds) in hyperspectral scenes, as a companion
to the coarser-resolution, column-retrieval-based Sentinel-5P methane
mapping done outside GRASS by the ALMA Energy toolset's
`05_methane_hotspots` script.

## NOTES

**This module computes a matched-filter enhancement statistic, not
necessarily a radiometrically calibrated ppm-m column retrieval.** Its
output is directly comparable in *relative* terms across a scene (larger
alpha = stronger CH4 signature), and is substantially more sensitive and
robust to background surface variability than a simple two-shoulder
band-depth ratio, because it exploits the full spectral window and the
scene's own background covariance rather than three isolated bands. If
*absorption_spectrum* supplies a unit absorption spectrum in properly
scaled physical units (e.g. from a full radiative-transfer model, as used
operationally by EMIT's own CH4 plume product), *alpha* approximates a
ppm-m enhancement; with the default synthetic Gaussian template it should
be treated as a relative ranking statistic only.

Because the retrieval whitens against the scene's own background
covariance, it is more susceptible to including plume-affected pixels
within the "background" mean/covariance estimate for scenes with very
large or pervasive plumes; for such cases consider excluding known plume
regions before running the module, or iterating (mask flagged hotspots,
recompute).

The module expects the input 3D raster to carry per-band wavelength
metadata, following the same convention as the rest of the *i.hyper*
family (an *i.hyper.import* `hyper.json` sidecar, or **wavelength**/
**unit** in each band's `r3.support` history). If no band lies within
*tolerance* nanometers of *absorption_wavelength*, or if fewer than 5
bands fall within the *left_shoulder*-*right_shoulder* window, the module
fails with `G_fatal_error` rather than silently proceeding with
insufficient spectral coverage -- this typically means the input's SWIR
coverage does not reach the CH4 absorption region near 2100-2400 nm.

## EXAMPLE

Compute a methane matched-filter enhancement from a hyperspectral cube
imported by *i.hyper.import*, using the default synthetic absorption
template:

```sh
i.hyper.methane input=emit_scene output=emit_ch4_mf
```

Using a real unit absorption spectrum and a narrower matched-filter
window:

```sh
i.hyper.methane input=emit_scene output=emit_ch4_mf \
    absorption_spectrum=ch4_unit_absorption.txt \
    left_shoulder=2150 right_shoulder=2350
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
