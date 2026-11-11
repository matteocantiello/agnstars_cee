# `agn_cee`: black-hole mergers inside stars in AGN disks

Code and data to reproduce

> **Signatures of Compact Object Mergers Inside Stars in AGN Disks**
> Cantiello et al. (2026)

## What the paper is about

Gas disks around supermassive black holes (active galactic nuclei, AGN) can feed stars so quickly that fresh
hydrogen reaches the stellar core as fast as it burns. These stars stop evolving and grow to hundreds of solar
masses; we call them **immortal stars**. The paper asks what happens when a stellar-mass black hole (BH), or a
pair of them, falls into such a star:

- a single BH sinks to the center in about a month;
- a pair of BHs at the center is driven together by the dense gas in about an hour, instead of the ~10⁵ years
  it would take through gravitational-wave emission alone;
- the merger looks like an ordinary LIGO–Virgo–KAGRA (LVK) event, but the gas leaves a strong imprint at
  deci-Hz frequencies, a band future detectors such as DECIGO would cover;
- the energy released heats the star by about its own binding energy, so each star hosts only one or two mergers.

The calculations are analytic and order-of-magnitude: they combine a stellar model with standard formulae for
gas drag, accretion and gravitational waves. Nothing here needs a supercomputer.

## Quick start

```bash
pip install -r requirements.txt
jupyter lab reproduce_paper.ipynb        # then: Kernel -> Restart & Run All
```

Run everything from the repository root, where the code expects the `data/` folder. The notebook takes
about 15 seconds on a laptop. It:

1. rebuilds the stellar model,
2. remakes all nine paper figures into `figures/` (same file names as in the manuscript), and
3. recomputes **77 numbers quoted in the paper** and checks each against the printed value. The last cell
   prints a table of all of them and stops with an error if any disagree.

Every step is explained in plain language, and a glossary at the top defines the technical terms.

## What is in this repository

```
.
├── reproduce_paper.ipynb   <- start here: every figure and every key number, explained
├── make_figures.py         regenerate all paper figures from the command line
├── agn_cee/                the Python package with all the physics
├── data/                   the stellar model and the LVK noise curve (see "Data")
├── figures/                the figures as used in the paper
├── tests/                  automated checks of the physics (pytest)
├── scripts/                an optional cross-check that needs external code (see below)
├── extras/                 a development notebook (optional; not needed to reproduce the paper)
├── requirements.txt
└── LICENSE
```

### Other ways to run things

| Task | Command |
|---|---|
| Regenerate the figures without Jupyter | `python make_figures.py` |
| Run the automated tests (30 tests, a few seconds) | `pytest tests/` |
| Compare our drag with the stratified-star calculation of Gagnier (2026) | see [below](#optional-cross-check-with-gagnier-2026) |

## Which file is which figure

The file names are the ones used in the LaTeX source; the first column is the figure number in the compiled paper.

| Paper | File | What it shows | Made by |
|---|---|---|---|
| Fig. 1 | `fig1_structure.pdf` | the immortal star and its accretion stream | `figures.fig_structure` |
| Fig. 2 | `fig3_caseA_inspiral.pdf` | how long a single BH takes to sink | `figures.fig_inspiral` |
| Fig. 3 | `fig9_spiral_trajectories.pdf` | the actual spiral paths | `review.fig_true_trajectories` |
| Fig. 4 | `fig2_caseA_powers.pdf` | energy lost to each drag channel (single BH) | `figures.fig_power` |
| Fig. 5 | `fig4_caseB_powers.pdf` | gas drag vs gravitational waves for a BH pair | `review.fig_power_caseB` |
| Fig. 6 | `fig8_energy_deposition.pdf` | energy the BH pair dumps into the star | `review.fig_energy_deposition` |
| Fig. 7 | `fig6_dephasing.pdf` | wave cycles lost relative to a merger in empty space | `figures.fig_dephasing` |
| Fig. 8 | `fig7_characteristic_strain.pdf` | signal strength vs detector sensitivity | `review.fig_characteristic_strain` |
| Fig. 9 (App. B) | `fig5_hardening_ladder.pdf` | the six gas-drag prescriptions compared | `review.fig_hardening_ladder` |

`figures/` also holds two supplementary figures that are not in the paper: `fig_orbit_accumulation.pdf` (where
the binary's orbits pile up, made by `extras/true_inspiral_trajectories.ipynb`) and `fig_unequal_mass_strain.pdf`
(unequal-mass binaries, made by `review.fig_unequal_mass_strain`).

## Data

- **`data/14547_cs10_RHO_3d-17_profile20.data`** is the stellar model: a 773 M☉ immortal star computed with the
  MESA stellar-evolution code, from the AGN-star models of Cantiello, Jermyn & Lin (2021). It lists density,
  temperature, pressure, composition and enclosed mass from the center to the surface. The code reads it directly
  and attaches an analytic accretion stream outside the surface. The MESA input files that produced it are
  archived separately in the MESA Zenodo community (doi:10.5281/zenodo.5071458).
- **`data/O3-L1-C01_..._sensitivity_strain_asd.txt`** is a representative noise curve of the LIGO Livingston
  detector during the third observing run (O3), used for the LVK signal-to-noise ratios. The LISA and DECIGO
  noise curves are analytic formulae from Robson, Cornish & Liu (2019) and Yagi & Seto (2011), coded in
  `agn_cee/detectors.py`.

## The `agn_cee` package

All quantities are in CGS units (grams, centimetres, seconds).

| Module | What it does |
|---|---|
| `structure.py` | builds the stellar model: the MESA star plus the accretion stream, with density, sound speed and enclosed mass as functions of radius, and the star's binding energies |
| `physics.py` | the basic formulae: dynamical-friction drag, accretion rate and accretion drag, gravitational-wave power, frequency and strain, Eddington limits |
| `inspiral.py` | a single BH sinking from the surface to the core |
| `bbh.py` | a BH pair shrinking in the core: gas drag (six prescriptions) vs gravitational waves, merger time, dephasing |
| `detectors.py` | LISA, DECIGO and LVK noise curves |
| `observability.py` | signal-to-noise ratio and time spent in each detector band |
| `rates.py` | the order-of-magnitude event-rate estimate (paper Sec. 4) |
| `figures.py`, `review.py` | build the paper figures; `review.py` also holds the robustness checks |
| `plotstyle.py` | shared figure style (AAS column widths, colour-blind-safe palette) |
| `constants.py` | physical constants |

A few modelling choices worth knowing before you change anything:

- **Sound speed.** The fiducial sound speed counts only gas pressure, $c_s=\sqrt{(5/3)P_{\rm gas}/\rho}$. The star
  is supported mostly by radiation pressure, and including it roughly doubles $c_s$ in the core. The paper and
  the notebook show that this lengthens the binary's merger from ~1 hr to ~4 hr without changing any conclusion.
- **Gas drag on a binary is uncertain.** `bbh.py` offers six prescriptions (the "ladder" of paper Appendix B).
  Model 2 is the fiducial one and the default everywhere.
- **Enclosed mass.** The code uses the MESA mass coordinate directly. Integrating $4\pi r^2\rho\,dr$ instead
  gives values 10–20% too high near the center.

## Optional: cross-check with Gagnier (2026)

The paper notes that drag in a stratified star exceeds the uniform-medium drag we use by a factor ~1–3, so our
sinking times are conservative. This needs Gagnier's public code, which is not bundled here:

```bash
git clone https://github.com/dgagnier/DF_stratified
python scripts/gagnier_stratified_check.py --dfstrat ./DF_stratified
```

## Software versions

Python ≥ 3.9 with NumPy ≥ 2.0, SciPy and Matplotlib (see `requirements.txt`). The paper figures were made with
Python 3.10, NumPy 2.2.5, SciPy 1.13.1 and Matplotlib 3.9.0. The notebook and tests also pass with
Python 3.12, NumPy 2.5, SciPy 1.18 and Matplotlib 3.11. The optional `mesa_reader` package is used if present;
otherwise a built-in reader loads the MESA file.

Regenerated figures match the committed ones curve for curve. Their PDF bounding box can differ by a few points
from machine to machine, because the tight crop depends on font rendering.

## How to cite

If you use this code, please cite the paper (Cantiello et al. 2026) and this software deposit:
doi:10.5281/zenodo.XXXXXXX.

## License

MIT; see `LICENSE`.
