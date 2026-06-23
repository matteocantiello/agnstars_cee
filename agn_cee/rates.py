"""Event-rate estimates for immortal-channel BBH mergers.

Implements the order-of-magnitude chain of the draft (Section "Rates"), kept
transparent and adjustable, and combines the volumetric rate with the detector
horizons from ``observability`` to get expected detection rates.

Chain
-----
1. Type I migration time of the immortal (Tanaka et al. 2004 scaling).
2. Encounter rate  Gamma = N_bh(<r0) / tau_mig.
3. BBH mergers per AGN per Myr, n_BBH (1-2 conservative, ~20 if all interior BH
   are processed before the star is unbound).
4. Volumetric rate  R = n_GN * f_AGN * n_BBH  [Gpc^-3 yr^-1].
5. Detection rate  N_det = R * V(D_horizon)  per detector.
"""

import numpy as np

from . import constants as cst

MYR = 1e6 * cst.YR
GPC = 1e3 * cst.MPC


def migration_time(M_star=1e2 * cst.MSUN, r0_over_rg=1e4, N=3.0, h=1e-3,
                   Sigma0=1e2, M8=1.0):
    """Type I migration time [s] for an embedded mass (draft eq.; Tanaka+2004).

    tau ~ 0.5 Myr (N/3)^-1 (r0/1e4 rg)^-1/2 (M/1e2 Msun)^-1 (h/1e-3)^2
                  (Sigma0/1e2)^-1 (M_SMBH/1e8 Msun)^3/2.
    Strongly sensitive to the disk aspect ratio h (tau ~ h^2).
    """
    return (0.5 * MYR * (N / 3.0) ** -1 * (r0_over_rg / 1e4) ** -0.5
            * (M_star / (1e2 * cst.MSUN)) ** -1 * (h / 1e-3) ** 2
            * (Sigma0 / 1e2) ** -1 * M8 ** 1.5)


def encounter_rate(N_bh_interior=10.0, **kw):
    """Immortal-BH encounter rate [s^-1] = N_bh(<r0) / tau_mig."""
    return N_bh_interior / migration_time(**kw)


def volumetric_rate(n_bbh_per_AGN_per_Myr, n_GN=4e-3, f_AGN=0.1):
    """Volumetric merger rate [Gpc^-3 yr^-1].

    n_GN : number density of host galaxies [Mpc^-3] (Baldry+2012, >= MW mass).
    f_AGN: active fraction. n_bbh: BBH mergers per AGN per Myr.
    """
    n_GN_Gpc3 = n_GN * 1e9                       # Mpc^-3 -> Gpc^-3
    n_bbh_per_yr = n_bbh_per_AGN_per_Myr / 1e6
    return n_GN_Gpc3 * f_AGN * n_bbh_per_yr


def detection_rate(R_vol, D_horizon, V_max_Gpc3=None):
    """Detection rate [yr^-1] = R_vol * (4/3 pi D^3).

    ``D_horizon`` in cm (use the sky-averaged SNR=8 distance, which approximates
    the detection-weighted range radius). ``V_max_Gpc3`` caps the volume at a
    cosmological ceiling when the horizon exceeds the regime where the
    non-redshifted SNR is valid.
    """
    D_Gpc = D_horizon / GPC
    V = (4.0 / 3.0) * np.pi * D_Gpc ** 3
    if V_max_Gpc3 is not None:
        V = min(V, V_max_Gpc3)
    return R_vol * V


# Comoving-volume ceilings (Planck18) used to cap super-horizon detectors.
V_COMOVING = {"z<0.5": 30.0, "z<1": 150.0, "z<2": 600.0}   # Gpc^3
