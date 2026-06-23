"""agn_cee: compact objects encountering immortal stars in AGN disks.

Clean, tested consolidation of the physics previously scattered across the
exploratory notebooks (CEE_AGN.ipynb, GWAnalytics.ipynb, paper_figures.ipynb).

Submodules
----------
constants : CGS physical constants.
physics   : drag, accretion, gravitational-wave, and orbital-energy formulae.
structure : MESA immortal-star profile + analytic accretion-stream model.
inspiral  : single compact-object quasi-circular inspiral into the star.
bbh       : binary-black-hole hardening and gravitational-wave dephasing.
"""

from . import constants, physics, structure, inspiral, bbh  # noqa: F401
