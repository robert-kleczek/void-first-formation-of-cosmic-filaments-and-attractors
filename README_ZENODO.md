# DRND Void-First Filament and Attractor Formation Preprint Package

Generated: 2026-06-07
Version: v.1.0.0
GitHub repository name: `void-first-formation-of-cosmic-filaments-and-attractors`

This package contains a reproducible technical-preprint draft testing a DRND-motivated void-first formation concept for cosmic filaments and attractors against public cosmic-web, void-lensing, and peculiar-velocity data.

The directory is organized in a Zenodo-compatible archival format, but the manuscript itself is not framed as a publication specific to Zenodo. It can be submitted, archived, or circulated through any appropriate research channel.

## Contents

- `paper/main.tex` - LaTeX source.
- `paper/main.pdf` - compiled PDF after running `xelatex`.
- `paper/orcid.png` - ORCID icon asset kept with the manuscript source.
- `paper/figures/` - figures used in the manuscript.
- `data/` - processed evidence tables and JSON summaries.
- `scripts/` - analysis/audit scripts used for velocity input checking and refined dynamics.
- `DATA_SOURCES.md` - raw public data sources and citations.
- `zenodo_metadata.json` - optional Zenodo-compatible metadata.

## Main Results

- The paper formulates filaments as secondary compression ridges generated after the birth and growth of void domains.
- Attractors are treated experimentally as convergence nodes where multiple void-shell fronts stall and concentrate matter.
- Earlier void-growth, void-in-cloud/void-in-void, Dipole-Repeller, and topological-skeleton results are treated as prior observational and methodological context, not as DRND discoveries.
- JWST high-redshift protostructures are explicitly separated from mature void-bounded filaments, so local quasar/protocluster strands are not treated as automatic falsifications.
- SDSS filament axes show a statistically robust radial-axis excess relative to the compensated void shell.
- The locked DRND void-lensing amplitude is within 0.5 per cent of the same-geometry free amplitude and strongly beats a zero-signal null by BIC.
- Carrick/2M++ vector-field monopoles show conditional outflow only for void-in-void environments.
- Strict line-of-sight PV tests using SDSS PV and Cosmicflows-4 do not recover a positive outflow signal.

## Rebuild PDF

From `paper/`:

```powershell
xelatex main.tex
xelatex main.tex
```

The LaTeX source uses the `orcidlink` package for the author ORCID marker.

The manuscript is deliberately written as a cautious observational technical note. It presents a falsifiable void-first chronology and does not claim that Lambda-CDM is already falsified.

## Falsification Boundary

The model does not predict the absence of every early filament-like protostructure. It predicts the delayed emergence of the mature, global, void-bounded filament-and-attractor architecture. A robust observation of an SDSS-like void-and-filament foam at `z > 6`, with large voids and thin separating ridges on tens-of-Mpc scales, would strongly damage the void-first chronology.
