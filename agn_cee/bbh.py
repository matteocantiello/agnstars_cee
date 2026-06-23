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

Validity caveat (the paper's central point): the point-mass-in-fixed-background
picture breaks down once the gas mass enclosed within the binary orbit becomes
comparable to the BBH mass. ``a_structure_uncertain`` returns that separation;
results at larger ``a`` are order-of-magnitude only.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import constants as cst
from . import physics


def _gas_loss_rate(m1, m2, a, rho, cs, C_d=1.0):
    """Energy-loss rate of the internal orbit to gas drag (DF + accretion) [erg/s].

    Each component i moves about the centre of mass at speed v_i = v_rel m_j/M
    on an orbit of radius r_i = a m_j/M; the drag power summed over components is
    returned.
    """
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)
    L = np.zeros_like(np.asarray(a, dtype=float))
    for mi, mj in ((m1, m2), (m2, m1)):
        vi = v_rel * mj / M
        ri = a * mj / M
        F_df = physics.dynamical_friction_force(mi, vi, ri, cs, rho, C_d=C_d)
        F_acc = physics.accretion_drag_force(rho, mi, cs, vi)
        L = L + (F_df + F_acc) * vi
    return L


def hardening_rates(m1, m2, a, rho, cs, C_d=1.0):
    """Return da/dt from GW and from gas, plus diagnostics, at separation ``a``.

    All rates are negative (hardening). Returns a dict.
    """
    a = np.asarray(a, dtype=float)
    L_gw = physics.gw_luminosity(m1, m2, a)
    L_gas = _gas_loss_rate(m1, m2, a, rho, cs, C_d=C_d)
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


def merger_time(m1, m2, a0, rho, cs, a_end=None, C_d=1.0, n=6000):
    """Merger time with gas vs in vacuum, integrating da/(da/dt) from a0 to a_end.

    Returns dict with t_gas, t_vacuum (= Peters), and their ratio.
    """
    a = _a_grid(m1, m2, a0, a_end, n)
    h = hardening_rates(m1, m2, a, rho, cs, C_d=C_d)
    dtda_gas = 1.0 / np.abs(h["dadt_tot"])     # dt/da
    dtda_vac = 1.0 / np.abs(h["dadt_gw"])
    t_gas = np.trapz(dtda_gas, a)
    t_vac = np.trapz(dtda_vac, a)
    return dict(a0=a0, t_gas=t_gas, t_vacuum=t_vac, ratio=t_gas / t_vac,
                t_peters=physics.t_merge_gw(m1, m2, a0))


def dephasing(m1, m2, a0, rho, cs, a_end=None, C_d=1.0, n=6000):
    """GW-cycle dephasing relative to vacuum, as a function of frequency.

    The number of GW cycles from separation a inward to merger is
    N(a) = integral f_gw dt = integral_{a_end}^{a} f_gw / |da/dt| da'. Gas drag
    speeds the chirp, so the binary spends fewer cycles between a given
    frequency and merger. The residual dephasing for a detector observing from
    frequency f to coalescence is

        dN(f) = N_vac(>f) - N_gas(>f)     [cycles],

    detectable when dN >~ O(0.1-1). Returns a, f_gw, N_vac, N_gas, dN and the
    total dephasing from a0 to merger.
    """
    a = _a_grid(m1, m2, a0, a_end, n)
    h = hardening_rates(m1, m2, a, rho, cs, C_d=C_d)
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
