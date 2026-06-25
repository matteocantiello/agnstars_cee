"""Binary-black-hole hardening and GW dephasing inside an immortal star (Case B).

A BBH (components m1, m2, internal separation a) is embedded at a location in the
star where the local gas has density ``rho`` and sound speed ``cs``. Two
separations matter and are kept distinct:

  * ``a``  -- the BBH internal separation, which sets the orbital velocity and
    the GW frequency;
  * the BBH's *location* in the star, which sets ``rho`` and ``cs`` (passed in).

The internal orbit hardens via GW emission and via gas drag (dynamical friction
+ accretion drag) acting on each component as it moves through the gas. For any
dissipative channel da/dt = -2 a^2 L / (G m1 m2), so the channels simply add.

The gas drag is described by a single "hardening ladder" (``gas_loss_rate_model``,
selected by ``model``; see ``HARDENING_MODELS``). This module is the one source of
truth for that prescription -- ``hardening_rates``, ``dephasing``, ``merger_time``
and everything downstream (observability SNR, the dephasing/energy figures) all
default to the fiducial ``model=2`` (nonlinear self-wake + KKSS08 companion wake).
The old independent-wake estimate is ``model=0`` (an upper bound on the drag).

Validity caveat (the paper's central point): the point-mass-in-fixed-background
picture breaks down once the gas mass enclosed within the binary orbit becomes
comparable to the BBH mass. ``a_structure_uncertain`` returns that separation;
results at larger ``a`` are order-of-magnitude only.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import constants as cst
from . import physics

FIDUCIAL_MODEL = 2   # nonlinear self-wake + KKSS08 companion wake


# --------------------------------------------------------------------------- #
# Gas-hardening "ladder" -- the single source of truth for the internal-binary
# gas drag. ``model=2`` (nonlinear self-wake + KKSS08 companion wake) is the
# fiducial prescription used throughout the paper; the other models bracket the
# systematic uncertainty (see HARDENING_MODELS) and are exercised in Appendix B.
# Model 0 is the old independent-wake baseline (self-wake DF + accretion, no
# companion term); it is an upper bound on the drag, not the fiducial.
# --------------------------------------------------------------------------- #
HARDENING_MODELS = {
    0: "independent wake / upper-drag baseline",
    1: "linear self + KKSS08 companion",
    2: "nonlinear self + KKSS08 (fiducial)",
    3: "CE wind-tunnel $C_d$",
    4: "shared-Bondi suppressed",
    5: "scaled companion (stress test)",
}


def kkss08_I2_phi(mach):
    """Companion-wake azimuthal drag coefficient I_{2,phi}(M), Kim, Kim & Sanchez-Salcedo 2008.

    Verbatim KKSS08 fit (accurate to ~6%). It is NEGATIVE -- opposite in sign to the self-wake
    azimuthal coefficient -- so the companion wake exerts a positive (forward) torque that
    partially cancels the backward self-wake drag. The net azimuthal coefficient is
    I_self + I_{2,phi} (do NOT take |I_{2,phi}|; its sign is the whole point).
    """
    mach = np.asarray(mach, dtype=float)
    val = np.where(mach < 2.97,
                   -0.022 * (10.0 - mach) * np.tanh(1.5 * mach),
                   -0.13 + 0.07 * np.arctan(5.0 * mach - 15.0))
    return val * mach**2


def companion_wake_fraction(mach, f_super=0.45):
    """Deprecated parametrization of the companion-wake cancellation; superseded by the
    verbatim ``kkss08_I2_phi`` fit. Retained for reference/tests."""
    mach = np.asarray(mach, dtype=float)
    eta = f_super + (0.90 - f_super) * np.clip(1.0 - mach, 0.0, 1.0)
    return np.clip(eta, 0.0, 0.95)


def cd_windtunnel(mach):
    """Common-envelope wind-tunnel drag coefficient C_d(M) (HL-normalised, uniform-rho).

    Representative bracket motivated by MacLeod et al. 2017 / De et al. 2020: C_d ~ a few,
    declining with Mach (their coefficients also depend on density gradient and EOS, here
    taken in the uniform limit). Used as F_drag = C_d * pi R_a^2 rho v^2.
    """
    mach = np.asarray(mach, dtype=float)
    return np.clip(3.0 / np.sqrt(np.maximum(mach, 0.3)), 0.7, 6.0)


def f_shared_bondi(chi_B, f_floor=0.1):
    """Gas-coupling suppression in the shared-Bondi / adiabatic regime (chi_B=R_B,BBH/a).

    Binary-Bondi simulations of adiabatic gas (e.g. self-limited-accretion preprint 2026)
    show entropy/pressure build-up suppresses accretion when the pair shares a Bondi sphere.
    Smoothly interpolate 1 (chi_B<<1) -> f_floor (chi_B>>1). Bracket only; exact value uncertain.
    """
    chi_B = np.asarray(chi_B, dtype=float)
    return f_floor + (1.0 - f_floor) / (1.0 + chi_B**2)


def _self_df_linear(m_p, v, a, cs, rho):
    """Linear (Coulomb-log only) self-wake DF force [dyn] -- Kim & Kim 2007 branch."""
    rmin = physics.canto_rmin(m_p, v)
    I = physics.coulomb_logarithm(v / cs, a, rmin)
    return I * 4.0 * np.pi * rho * (cst.G * m_p) ** 2 / v**2


def gas_loss_rate_model(m1, m2, a, rho, cs, model=FIDUCIAL_MODEL):
    """Gas energy-loss rate L_gas of the internal orbit for a given ladder model [erg/s].

    Each component i moves about the centre of mass at speed v_i = v_rel m_j/M on an orbit of
    radius r_i = a m_j/M; the drag power summed over components is returned. ``model`` selects the
    hardening prescription (see HARDENING_MODELS); the fiducial is ``model=2``.
    """
    a = np.asarray(a, dtype=float)
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)
    L = np.zeros_like(a)
    for mi, mj in ((m1, m2), (m2, m1)):
        vi = v_rel * mj / M
        ri = a * mj / M       # orbital radius about the c.o.m. (Coulomb-log length scale)
        mach = vi / cs
        F_acc = physics.accretion_drag_force(rho, mi, cs, vi)
        # KKSS08 companion-wake force (additive, sign preserved): F0 * I_{2,phi} < 0.
        F0 = 4.0 * np.pi * rho * (cst.G * mi) ** 2 / vi**2
        F_comp = F0 * kkss08_I2_phi(mach)
        if model == 0:                                   # independent self-wake (upper-drag baseline)
            F_df = physics.dynamical_friction_force(mi, vi, ri, cs, rho)
        elif model == 1:                                 # linear self + KKSS08 companion
            F_df = _self_df_linear(mi, vi, ri, cs, rho) + F_comp
        elif model == 2:                                 # nonlinear self + KKSS08 companion (fiducial)
            F_df = physics.dynamical_friction_force(mi, vi, ri, cs, rho) + F_comp
        elif model == 3:                                 # CE wind-tunnel C_d (incl. accretion)
            Ra = physics.bondi_hoyle_radius(mi, vi, cs)
            F_df = cd_windtunnel(mach) * np.pi * Ra**2 * rho * vi**2
            F_acc = 0.0
        elif model == 4:                                 # shared-Bondi suppressed (applied to model 2)
            F_df = physics.dynamical_friction_force(mi, vi, ri, cs, rho) + F_comp
        elif model == 5:                                 # scaled companion (stress test, NOT fiducial)
            # fractional cancellation: assumes the linear companion/self ratio survives into the
            # nonlinear regime (a strong extrapolation). "Maximal companion cancellation" bound.
            F_df = physics.dynamical_friction_force(mi, vi, ri, cs, rho) * (1.0 - companion_wake_fraction(mach))
        else:
            raise ValueError(f"unknown hardening model {model}")
        L = L + (F_df + F_acc) * vi
    if model == 4:
        L = L * f_shared_bondi(cst.G * M / (cs**2 * a))
    # Floor at 0: where the (linear) companion wake over-cancels a weak self-wake the net
    # gas torque turns "forward" (apparent expansion) -- a linear-theory artifact in the
    # supersonic core (R_BHL/r >> 1). We conservatively take gas coupling = 0 there and let
    # GW drive. This affects only Model 1 (linear self); Models 0,2,3,4 never go negative.
    return np.clip(L, 0.0, None)


def dadt_model(m1, m2, a, rho, cs, model=FIDUCIAL_MODEL):
    """Return (da/dt_total, da/dt_GW) for a ladder model [cm/s]."""
    L_gas = gas_loss_rate_model(m1, m2, a, rho, cs, model)
    L_gw = physics.gw_luminosity(m1, m2, a)
    pref = 2.0 * a**2 / (cst.G * m1 * m2)
    return -pref * (L_gas + L_gw), -pref * L_gw


def hardening_rates(m1, m2, a, rho, cs, model=FIDUCIAL_MODEL):
    """Return da/dt from GW and from gas, plus diagnostics, at separation ``a``.

    Gas drag uses the hardening ladder ``gas_loss_rate_model(..., model)``; the default is the
    fiducial Model 2 (nonlinear self-wake + KKSS08 companion). All rates are negative
    (hardening). Returns a dict.
    """
    a = np.asarray(a, dtype=float)
    L_gw = physics.gw_luminosity(m1, m2, a)
    L_gas = gas_loss_rate_model(m1, m2, a, rho, cs, model=model)
    pref = 2.0 * a**2 / (cst.G * m1 * m2)
    dadt_gw = -pref * L_gw
    dadt_gas = -pref * L_gas
    v_rel = physics.v_kepler(m1, m2, a)
    return dict(a=a, L_gw=L_gw, L_gas=L_gas,
                dadt_gw=dadt_gw, dadt_gas=dadt_gas, dadt_tot=dadt_gw + dadt_gas,
                v_rel=v_rel, mach=(v_rel * m2 / (m1 + m2)) / cs,
                f_gw=physics.gw_frequency(m1, m2, a))


def a_structure_uncertain(m1, m2, rho):
    """Separation at which the enclosed gas mass equals the BBH mass [cm].

    M_gas(<a) = rho (4/3) pi a^3 = (m1+m2)  ->  a_crit. Above this the binary
    actively restructures its surroundings and the analytic treatment is only
    order-of-magnitude.
    """
    return (3.0 * (m1 + m2) / (4.0 * np.pi * rho)) ** (1.0 / 3.0)


def a_isco(m1, m2):
    """Crude ISCO separation 6 G M / c^2 used as the inner integration cutoff."""
    return 6.0 * cst.G * (m1 + m2) / cst.C**2


def _a_grid(m1, m2, a0, a_end=None, n=6000):
    if a_end is None:
        a_end = a_isco(m1, m2)
    return np.logspace(np.log10(a_end), np.log10(a0), n)


def merger_time(m1, m2, a0, rho, cs, a_end=None, model=FIDUCIAL_MODEL, n=6000):
    """Merger time with gas vs in vacuum, integrating da/(da/dt) from a0 to a_end.

    Returns dict with t_gas, t_vacuum (= Peters), and their ratio.
    """
    a = _a_grid(m1, m2, a0, a_end, n)
    h = hardening_rates(m1, m2, a, rho, cs, model=model)
    dtda_gas = 1.0 / np.abs(h["dadt_tot"])     # dt/da
    dtda_vac = 1.0 / np.abs(h["dadt_gw"])
    t_gas = np.trapz(dtda_gas, a)
    t_vac = np.trapz(dtda_vac, a)
    return dict(a0=a0, t_gas=t_gas, t_vacuum=t_vac, ratio=t_gas / t_vac,
                t_peters=physics.t_merge_gw(m1, m2, a0))


def merger_time_model(m1, m2, a0, rho, cs, model=FIDUCIAL_MODEL, a_end=None, n=4000):
    """Merger time integrating da/(da/dt_total) from a0 to ISCO for a ladder model [s]."""
    if a_end is None:
        a_end = a_isco(m1, m2)
    a = np.logspace(np.log10(a_end), np.log10(a0), n)
    dadt_tot, _ = dadt_model(m1, m2, a, rho, cs, model)
    return float(np.trapz(1.0 / np.abs(dadt_tot), a))


def dephasing(m1, m2, a0, rho, cs, a_end=None, model=FIDUCIAL_MODEL, n=6000):
    """GW-cycle dephasing relative to vacuum, as a function of frequency.

    The number of GW cycles from separation a inward to merger is
    N(a) = integral f_gw dt = integral_{a_end}^{a} f_gw / |da/dt| da'. Gas drag
    speeds the chirp, so the binary spends fewer cycles between a given
    frequency and merger. The residual dephasing for a detector observing from
    frequency f to coalescence is

        dN(f) = N_vac(>f) - N_gas(>f)     [cycles],

    detectable when dN >~ O(0.1-1). Returns a, f_gw, N_vac, N_gas, dN and the
    total dephasing from a0 to merger. Gas hardening uses the fiducial Model 2.
    """
    a = _a_grid(m1, m2, a0, a_end, n)
    h = hardening_rates(m1, m2, a, rho, cs, model=model)
    f = h["f_gw"]

    # cycles per unit separation: dN/da = f / |da/dt|
    dNda_vac = f / np.abs(h["dadt_gw"])
    dNda_tot = f / np.abs(h["dadt_tot"])
    # N(>f): cycles from separation a down to a_end (merger), counted by
    # integrating from the inner edge a_end (=a[0]) outward to a.
    N_vac = cumulative_trapezoid(dNda_vac, a, initial=0.0)
    N_gas = cumulative_trapezoid(dNda_tot, a, initial=0.0)
    dN = N_vac - N_gas
    return dict(a=a, f_gw=f, N_vac=N_vac, N_gas=N_gas, dN=dN,
                dN_total=float(dN[-1]),
                a_uncertain=a_structure_uncertain(m1, m2, rho))
