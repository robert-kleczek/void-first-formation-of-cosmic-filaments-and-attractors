# Data Sources

The package includes processed summary tables and reproducibility scripts. Raw public data files are not all bundled, because the Carrick/2M++ cubes are large. The raw sources used were:

## SDSS void lensing

- Clampitt, J. and Jain, B. 2015, *Lensing measurements of the mass distribution in SDSS voids*, MNRAS 454, 3357-3365.
- DOI: https://doi.org/10.1093/mnras/stv2215
- Local processed inputs used in this project:
  - `REAL_SDSS_VOID_LENSING/real_sdss_void_catalog.csv`
  - `REAL_SDSS_VOID_LENSING/real_sdss_void_lensing_rebinned.csv`

## SDSS DR8 filament catalogue

- Tempel, E., Stoica, R. S. and Saar, E. 2014, *Detecting filamentary pattern in the cosmic web: a catalogue of filaments for the SDSS*, MNRAS 438, 3465-3482.
- DOI: https://doi.org/10.1093/mnras/stt2454
- Catalogue landing page used during analysis: http://www.aai.ee/~elmo/sdss-filaments/
- Local raw file:
  - `REAL_SDSS_FILAMENTS/dr8_filaments.fits`

## SDSS Peculiar Velocity Catalogue

- Howlett et al. 2022, *The SDSS Peculiar Velocity Catalogue*.
- Zenodo DOI: https://doi.org/10.5281/zenodo.6824749
- Local raw file:
  - `REAL_VELOCITY_FIELDS_20260606/sdss_pv/SDSS_PV_public.dat`

## Cosmicflows-4

- Tully et al. 2023, *Cosmicflows-4*, ApJ 944, 94.
- DOI: https://doi.org/10.3847/1538-4357/ac94d8
- VizieR catalogue: https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/944/94
- Local raw VizieR TSV files:
  - `REAL_VELOCITY_FIELDS_20260606/cosmicflows4_vizier/cf4_table2_individual_galaxies.tsv`
  - `REAL_VELOCITY_FIELDS_20260606/cosmicflows4_vizier/cf4_groups_peculiar_velocities.tsv`

## Carrick / 2M++ density and velocity fields

- Carrick, J., Turnbull, S. J., Lavaux, G. and Hudson, M. J. 2015, *Cosmological parameters from the comparison of peculiar velocities with predictions from the 2M++ density field*, MNRAS 450, 317-332.
- DOI: https://doi.org/10.1093/mnras/stv547
- Data portal: https://cosmicflows.iap.fr/download/
- Local raw files:
  - `REAL_VELOCITY_FIELDS_20260606/carrick_2mpp/twompp_density.npy`
  - `REAL_VELOCITY_FIELDS_20260606/carrick_2mpp/twompp_velocity.npy`

## SDSS/eBOSS RSD

- SDSS final BAO/RSD measurements page: https://www.sdss4.org/science/final-bao-and-rsd-measurements/
- eBOSS DR16 LRGxELG data repository used here: https://github.com/icosmology/eBOSS_DR16_LRGxELG

## High-redshift context for falsification

These sources are cited in the revised manuscript as context for the high-redshift ``kill switch'' of the void-first chronology. They are not used in the local SDSS/2M++ numerical tests bundled here.

- Wang et al. 2023, *ASPIRE: JWST Reveals a Filamentary Structure around a z=6.61 Quasar*, ApJL 951, L4. DOI: https://doi.org/10.3847/2041-8213/accd6f
- Hatamnia et al. 2026, *Large-scale Structure in COSMOS-Web: Tracing Galaxy Evolution in the Cosmic Web up to z~7 with the Largest JWST Survey*, ApJ 1002, 192. DOI: https://doi.org/10.3847/1538-4357/ae5bac
- Stark et al. 2015, *Finding high-redshift voids using Lyman alpha forest tomography*, MNRAS 453, 4311-4323. DOI: https://doi.org/10.1093/mnras/stv1868
- Krolewski et al. 2018, *A Detection of z~2.3 Cosmic Voids from 3D Lyman-alpha Forest Tomography in the COSMOS Field*. DOI: https://doi.org/10.3847/1538-4357/aac829

## Prior-work context for novelty boundary

These sources are cited to separate established void/cosmic-web phenomenology from the proposed DRND causal mechanism. They are not used as numerical inputs in the bundled tests.

- Sheth and van de Weygaert 2004, *A hierarchy of voids: much ado about nothing*, MNRAS 350, 517-538. DOI: https://doi.org/10.1111/j.1365-2966.2004.07661.x
- Hoffman et al. 2017, *The Dipole Repeller*, Nature Astronomy 1, 0036. DOI: https://doi.org/10.1038/s41550-016-0036
- Sousbie 2011, *The persistent cosmic web and its filamentary structure - I. Theory and implementation*, MNRAS 414, 350-383. DOI: https://doi.org/10.1111/j.1365-2966.2011.18394.x
- Ambjorn, Jurkiewicz and Loll 2005, *Spectral dimension of the universe*, Physical Review Letters 95, 171301. DOI: https://doi.org/10.1103/PhysRevLett.95.171301
