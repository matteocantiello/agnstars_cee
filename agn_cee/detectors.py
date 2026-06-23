"""Sky-averaged GW detector noise curves (one-sided strain PSD S_n(f), units 1/Hz).

* LISA   : analytic sky-averaged sensitivity of Robson, Cornish & Liu 2019
           (arXiv:1803.01944 = CQG 36, 105011), Eq. (1), with the single-link optical
           metrology and acceleration noises of Eqs. (10)-(11) and the 4-yr galactic
           confusion foreground of Eq. (14) / Table 1.
* DECIGO : analytic noise PSD of Yagi & Seto 2011 (PRD 83, 044011), Eq. (5).
* LVK    : O3 representative strain ASD shipped in data/ (S_n = ASD^2).

Each ``Sn_*`` returns S_n(f) and is set to +inf outside the instrument band, so
1/S_n -> 0 and the band is enforced automatically inside SNR integrals.
``BANDS`` gives nominal (f_lo, f_hi) for time-in-band bookkeeping and plotting.
"""

import os

import numpy as np
from scipy.interpolate import interp1d

from . import constants as cst

BANDS = {            # nominal sensitive bands [Hz]
    "LISA": (1e-4, 1e-1),
    "DECIGO": (1e-2, 1e1),
    "LVK": (1e1, 1e3),
}

# Sky/inclination/polarization averaging factor applied to the matched-filter SNR.
# The LISA (Robson+2019) and DECIGO (Yagi & Seto 2011) curves are *sensitivities* with the
# averaged signal response R(f) already divided in, so SNR(optimal amplitude, averaged Sn) is
# already sky-averaged -> factor 1. The LVK O3 curve is the *raw* single-detector noise ASD, so
# with the face-on (optimal) source amplitude the SNR is optimally-oriented; the single-detector
# sky+inclination+polarization average is 2/5 = sqrt(4/25) (<F+^2>=<Fx^2>=1/5). Validated against
# the O3 BNS range (~120 Mpc) and GW150914/GW170608 (see tests).
SKY_AVG = {"LISA": 1.0, "DECIGO": 1.0, "LVK": 0.4}


# --------------------------------------------------------------------------- #
# LISA  (Robson, Cornish & Liu 2019)
# --------------------------------------------------------------------------- #
def Sn_lisa(f, confusion=True, Tobs_yr=4.0):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    L = 2.5e9                       # arm length [m]
    fstar = cst.C / 100.0 / (2 * np.pi * L)   # c in m/s = C/100 (C is cm/s); ~19.09 mHz
    # single-link noises: Robson, Cornish & Liu (2019) Eqs. (10)-(11)
    P_oms = (1.5e-11) ** 2 * (1 + (2e-3 / f) ** 4)                 # m^2/Hz, Eq. (10)
    P_acc = (3e-15) ** 2 * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)  # m^2 s^-4 /Hz, Eq. (11)
    # Sky-averaged sensitivity, Robson+2019 Eq. (1): the [1 + 0.6 (f/f*)^2] factor is
    # the approximate averaged response; the acceleration term carries the constant 4.
    Sn = (10.0 / (3 * L ** 2)) * (P_oms + 4.0 * P_acc / (2 * np.pi * f) ** 4) \
        * (1 + 0.6 * (f / fstar) ** 2)
    if confusion:
        Sn = Sn + _lisa_confusion(f, Tobs_yr)
    Sn = np.where((f > 1e-5) & (f < 1.0), Sn, np.inf)
    return Sn


def _lisa_confusion(f, Tobs_yr):
    # Robson+2019 Table 1 fit; coefficients for Tobs = 4 yr.
    A = 9e-45
    a, b, k, g, fk = 0.138, -221.0, 521.0, 1680.0, 0.00113
    with np.errstate(over="ignore", invalid="ignore"):
        Sc = A * f ** (-7.0 / 3.0) * np.exp(-f ** a + b * f * np.sin(k * f)) \
            * (1 + np.tanh(g * (fk - f)))
    return np.nan_to_num(Sc, nan=0.0, posinf=0.0)


# --------------------------------------------------------------------------- #
# DECIGO  (Yagi & Seto 2011, eq. 5)
# --------------------------------------------------------------------------- #
def Sn_decigo(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    fp = 7.36
    Sn = (7.05e-48 * (1 + (f / fp) ** 2)
          + 4.8e-51 * f ** -4 / (1 + (f / fp) ** 2)
          + 5.33e-52 * f ** -4)
    return np.where((f > 1e-3) & (f < 1e2), Sn, np.inf)


# --------------------------------------------------------------------------- #
# LVK  (O3 representative ASD)
# --------------------------------------------------------------------------- #
_O3_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt")
_o3 = np.loadtxt(_O3_PATH)
_o3_f, _o3_asd = _o3[:, 0], _o3[:, 1]
_o3_logSn = interp1d(np.log10(_o3_f), np.log10(_o3_asd ** 2),
                     bounds_error=False, fill_value=np.inf)


def Sn_lvk(f):
    f = np.atleast_1d(np.asarray(f, dtype=float))
    out = 10.0 ** _o3_logSn(np.log10(f))
    return np.where((f >= _o3_f[0]) & (f <= _o3_f[-1]), out, np.inf)


DETECTORS = {"LISA": Sn_lisa, "DECIGO": Sn_decigo, "LVK": Sn_lvk}
