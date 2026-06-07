# Void-First Formation of Cosmic Filaments and Attractors

Version: v.1.0.0

Repository name: `void-first-formation-of-cosmic-filaments-and-attractors`

This repository contains a reproducible technical-preprint package testing a DRND-motivated void-first formation concept for cosmic filaments and attractors against public cosmic-web, void-lensing, and peculiar-velocity data.

## Contents

- `paper/main.tex` - LaTeX source for the manuscript.
- `paper/main.pdf` - compiled manuscript PDF.
- `paper/orcid.png` - ORCID icon asset kept with the manuscript source.
- `paper/figures/` - manuscript figures.
- `data/` - processed evidence tables and JSON summaries.
- `scripts/` - analysis and audit scripts.
- `DATA_SOURCES.md` - raw public data sources and citations.
- `DRND_II_ZENODO_ABSTRACT.md` - repository/deposit abstract text.
- `README_ZENODO.md` - Zenodo-compatible archival notes.
- `zenodo_metadata.json` - optional Zenodo-compatible metadata.
- `SHA256_MANIFEST.csv` - file integrity manifest.

## Main Claim Under Test

The paper formulates filaments as secondary compression ridges generated after the birth and growth of void domains. Attractors are treated experimentally as convergence nodes where multiple void-shell fronts stall and concentrate matter.

The manuscript is deliberately written as a cautious observational technical note. It presents a falsifiable void-first chronology and does not claim that Lambda-CDM is already falsified.

## Rebuild PDF

From `paper/`:

```powershell
xelatex main.tex
xelatex main.tex
```

The LaTeX source uses the `orcidlink` package for the author ORCID marker.
