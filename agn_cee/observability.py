"""Matched-filter SNR and time-in-band for a BBH chirping inside an immortal star.

The sky-averaged matched-filter SNR is

    SNR^2 = 4 \\int |h~(f)|^2 / S_n(f) df = \\int A(f)^2 / (f_dot S_n(f)) df,

where A(f) is the angle-averaged strain amplitude (the same amplitude returned by
``physics.gw_strain``) and f_dot is the *actual* chirp rate. Changing variables
from f to the binary separation a (f_dot = |df/da| |da/dt|) gives the compact form

    SNR^2 = \\int A(a)^2 / (|da/dt| S_n(f(a))) da,

so the only difference between a vacuum inspiral and one hardened by gas is |da/dt|
(GW-only vs GW+gas). Because gas makes |da/dt| much larger, the binary spends less
time -- fewer cycles -- in band, and the SNR is suppressed accordingly. This is the
quantitative version of "the immortal-channel binary plunges through the LISA band."
"""

import numpy as np

from . import constants as cst
from . import physics
from . import bbh
from . import detectors


def chirp(m1, m2, rho, cs, distance, gas=True, f_start=1e-5, a_end=None, n=8000):
    """Tabulate the inspiral: separation, frequency, |da/dt|, amplitude.

    ``gas=True`` uses GW+gas hardening at local (rho, cs); ``gas=False`` is the
    vacuum (GW-only) inspiral. ``distance`` in cm.
    """
    a_start = physics.separation_from_gw_frequency(m1, m2, f_start)
    if a_end is None:
        a_end = bbh.a_isco(m1, m2)
    a = np.logspace(np.log10(a_end), np.log10(a_start), n)
    h = bbh.hardening_rates(m1, m2, a, rho, cs)
    adot = np.abs(h["dadt_tot"] if gas else h["dadt_gw"])
    f, A = physics.gw_strain(m1, m2, a, distance)
    return dict(a=a, f=f, adot=adot, A=A)


def snr(m1, m2, rho, cs, distance, detector, gas=True, **kw):
    """Sky-averaged matched-filter SNR in a given detector (name or S_n callable).

    A(f) is the face-on (optimal) amplitude; the per-detector ``SKY_AVG`` factor converts the
    optimally-oriented SNR to the sky/inclination/polarization-averaged value (1 for LISA/DECIGO
    whose sensitivities already include the averaged response; 2/5 for the raw LVK ASD).
    """
    Sn_func = detectors.DETECTORS[detector] if isinstance(detector, str) else detector
    fac = detectors.SKY_AVG.get(detector, 1.0) if isinstance(detector, str) else 1.0
    c = chirp(m1, m2, rho, cs, distance, gas=gas, **kw)
    Sn = Sn_func(c["f"])
    integrand = c["A"] ** 2 / (c["adot"] * Sn)        # 1/cm; ->0 outside the band
    integrand = np.where(np.isfinite(integrand), integrand, 0.0)
    return fac * np.sqrt(np.trapz(integrand, c["a"]))


def time_in_band(m1, m2, rho, cs, band, gas=True, n=8000):
    """Time [s] the binary spends with f_GW inside ``band`` = (f_lo, f_hi)."""
    f_lo, f_hi = band
    a_lo = physics.separation_from_gw_frequency(m1, m2, f_hi)   # high f -> small a
    a_hi = physics.separation_from_gw_frequency(m1, m2, f_lo)
    a = np.logspace(np.log10(a_lo), np.log10(a_hi), n)
    h = bbh.hardening_rates(m1, m2, a, rho, cs)
    adot = np.abs(h["dadt_tot"] if gas else h["dadt_gw"])
    return float(np.trapz(1.0 / adot, a))                       # dt = da/|da/dt|


def horizon_distance(m1, m2, rho, cs, detector, gas=True, snr_thresh=8.0,
                     d_ref=100 * cst.MPC, **kw):
    """Distance [cm] at which the SNR equals ``snr_thresh`` (SNR ~ 1/distance)."""
    s_ref = snr(m1, m2, rho, cs, d_ref, detector, gas=gas, **kw)
    return d_ref * s_ref / snr_thresh
