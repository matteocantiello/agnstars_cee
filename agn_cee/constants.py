"""Physical constants in CGS units.

Ported from the original ``Constants.py`` (which printed a banner on import and
exposed the values as instance attributes of a ``Constants`` class). Here they
are plain module-level constants so callers can simply do ``from agn_cee import
constants as cst`` and use ``cst.G`` etc. The original attribute names are kept.
"""

# --- Fundamental / astronomical constants (CGS) ---
MSUN = 1.989e33        # solar mass [g]
RSUN = 6.955e10        # solar radius [cm]
LSUN = 3.827e33        # solar luminosity [erg/s]
G = 6.674e-8           # gravitational constant [cm^3 g^-1 s^-2]
YR = 3.1536e7          # Julian year [s]
H_PLANCK = 6.6260755e-27   # Planck constant [erg s]
KB = 1.380658e-16      # Boltzmann constant [erg/K]
MP = 1.6726219e-24     # proton mass [g]
ME = 9.10938356e-28    # electron mass [g]
C = 2.99792458e10      # speed of light [cm/s]
PC = 3.085677581e18    # parsec [cm]
MPC = 1e6 * PC         # megaparsec [cm]
AU = 1.496e13          # astronomical unit [cm]
QE = 4.8032068e-10     # elementary charge [esu]
EV = 1.6021772e-12     # electron volt [erg]
SIGMA_SB = 5.67051e-5  # Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
SIGMA_T = 6.6524e-25   # Thomson cross section [cm^2]
RGAS = 8.3145e7        # gas constant [erg K^-1 mol^-1]


class Constants:
    """Backwards-compatible shim: exposes the constants as attributes.

    The legacy notebooks do ``from Constants import Constants; c = Constants()``
    and then use ``c.G``, ``c.msun`` (lower case). This class reproduces that
    interface (without the print side effect) so old code keeps working.
    """

    def __init__(self):
        self.msun = MSUN
        self.rsun = RSUN
        self.lsun = LSUN
        self.G = G
        self.yr = YR
        self.h = H_PLANCK
        self.kB = KB
        self.mp = MP
        self.me = ME
        self.c = C
        self.pc = PC
        self.au = AU
        self.q = QE
        self.eV = EV
        self.sigmaSB = SIGMA_SB
        self.sigmaT = SIGMA_T
        self.Rg = RGAS
