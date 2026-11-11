"""agn_cee: black holes falling into and merging inside "immortal" stars in AGN disks.

Code behind Cantiello et al. (2026), "Signatures of Compact Object Mergers Inside
Stars in AGN Disks". Start with ``reproduce_paper.ipynb`` in the repository root,
which walks through every figure and number in the paper using this package.

Submodules
----------
constants     : physical constants (CGS units).
physics       : the basic formulae -- drag, accretion, gravitational waves, orbits.
structure     : the stellar model (MESA star + accretion stream from the AGN disk).
inspiral      : a single black hole sinking from the stellar surface to the core.
bbh           : a black-hole binary shrinking in the core (gas drag vs GW emission).
detectors     : LISA, DECIGO and LIGO-Virgo-KAGRA noise curves.
observability : signal-to-noise ratio and time spent in each detector band.
rates         : order-of-magnitude event-rate estimate.
figures       : builders for paper Figs. 1, 2, 4 and 7.
review        : builders for paper Figs. 3, 5, 6, 8 and 9, plus robustness checks.
plotstyle     : shared Matplotlib style.
"""

from . import constants, physics, structure, inspiral, bbh  # noqa: F401
