"""Single compact-object inspiral into an immortal star (Case A).

A compact object (CO) of mass ``m_co`` spirals through the star on a
quasi-circular orbit at radius ``r``, orbiting the enclosed mass
M_enc(r) = M_star(<r) + m_central (a central black hole, if present). The orbit
decays as the CO loses energy to gaseous dynamical friction, accretion drag and
gravitational waves.

We use the orbit-averaged angular-momentum-loss formulation, which is robust for
an extended (non-point) mass distribution. For any tangential dissipative force
the torque equals (power / orbital frequency), so

    dL_ang/dt = -(L_DF + L_acc + L_GW) / Omega,

with the circular-orbit angular momentum L_ang(r) = m_co sqrt(G M_enc r) and
Omega(r) = sqrt(G M_enc / r^3). This gives a strictly inward decay,

    dt/dr = -(dL_ang/dr) * Omega / L_total > 0  (since dr < 0),

which we integrate by quadrature to get the spiral-in time t(r0).
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import constants as cst
from . import physics


def channel_luminosities(model, m_co, r, m_central=0.0, C_d=1.0,
                         include_gw=True, include_acc=True, include_df=True):
    """Energy-loss rates and orbital kinematics for a CO at radius ``r`` (arrays).

    Returns a dict with v_c, mach, M_enc, and the per-channel and total
    luminosities [erg/s].
    """
    r = np.asarray(r, dtype=float)
    rho = model.rho_of_r(r)
    cs = model.cs_of_r(r)
    m_enc = np.clip(model.menc_of_r(r) + m_central, 1e-30, None)

    v_c = physics.v_circular(m_enc, r)
    mach = v_c / cs

    L_df = physics.dynamical_friction_luminosity(
        m_co, v_c, r, cs, rho, C_d=C_d) if include_df else np.zeros_like(r)
    L_acc = physics.accretion_drag_luminosity(
        rho, m_co, cs, v_c) if include_acc else np.zeros_like(r)
    # GW: treat (m_co, M_enc) as an effective binary (meaningful when M_enc is
    # centrally concentrated, e.g. a central BH). Subdominant otherwise.
    L_gw = physics.gw_luminosity(m_co, m_enc, r) if include_gw else np.zeros_like(r)

    L_total = L_df + L_acc + L_gw
    return dict(r=r, rho=rho, cs=cs, m_enc=m_enc, v_c=v_c, mach=mach,
                L_df=L_df, L_acc=L_acc, L_gw=L_gw, L_total=L_total)


def spiral_in_time(model, m_co, r0, r_final=None, m_central=0.0, C_d=1.0,
                   n=4000, **channels):
    """Spiral-in time and trajectory for a CO sinking from ``r0`` to ``r_final``.

    Parameters
    ----------
    model : StarStreamModel
    m_co : float
        Compact-object mass [g].
    r0 : float
        Initial (capture) radius [cm].
    r_final : float, optional
        Inner radius [cm]. Default: the radius where M_star(<r) = m_co (the CO
        reaches the core and locally dominates the gas mass).
    m_central : float
        Mass of a pre-existing central black hole [g].

    Returns
    -------
    dict with r (centre->out), t_to_r0 (time to fall from r0 down to r),
    t_total, and the channel luminosities along the way.
    """
    if r_final is None:
        # radius where the enclosed *stellar* mass equals the CO mass
        i = int(np.argmin(np.abs(model.m_enc - m_co)))
        r_final = max(model.r[i], model.r[1])
    r_final = max(r_final, model.r[1])

    r = np.logspace(np.log10(r_final), np.log10(r0), n)
    ch = channel_luminosities(model, m_co, r, m_central=m_central, C_d=C_d,
                              **channels)

    m_enc = ch["m_enc"]
    L_ang = m_co * np.sqrt(cst.G * m_enc * r)
    omega = np.sqrt(cst.G * m_enc / r**3)
    dLang_dr = np.gradient(L_ang, r)

    # dt/dr = (dL_ang/dr) * Omega / L_total  (time accumulated falling inward)
    integrand = dLang_dr * omega / ch["L_total"]
    t_to_r0 = cumulative_trapezoid(integrand, r, initial=0.0)  # time from r_final to r

    out = dict(r=r, t_to_r0=t_to_r0, t_total=float(t_to_r0[-1]),
               r_final=r_final, r0=r0, m_co=m_co, m_central=m_central)
    out.update({k: ch[k] for k in
                ("v_c", "mach", "m_enc", "rho", "cs",
                 "L_df", "L_acc", "L_gw", "L_total")})
    return out


def trajectory(sol, n_t=500):
    """Convert a spiral_in_time solution into r(t) (t measured from capture).

    The CO starts at r0 (t=0) and reaches r_final at t_total.
    """
    r = sol["r"]
    # time since capture = t_total - t_to_r0 (t_to_r0 is measured from r_final)
    t_since_capture = sol["t_total"] - sol["t_to_r0"]
    # sort by increasing time
    order = np.argsort(t_since_capture)
    t_sorted = t_since_capture[order]
    r_sorted = r[order]
    t_grid = np.linspace(0.0, sol["t_total"], n_t)
    r_of_t = np.interp(t_grid, t_sorted, r_sorted)
    return t_grid, r_of_t


def destruction_energetics(model, m_co, eta=0.1, kappa=0.34):
    """Order-of-magnitude budget for what happens once the CO reaches the core.

    Compares (i) the orbital energy released during the inspiral and
    (ii) the accretion energy available, to the stellar binding energy, and
    brackets the destruction timescale between the (feedback-free) Bondi rate
    and the Eddington-limited rate.
    """
    U_bind = abs(model.U_bind[0])                      # |total binding energy|
    rho_c, cs_c = model.rho_c, model.cs_of_r(model.r[1])

    # orbital energy released sinking to where M_enc = m_co
    i = int(np.argmin(np.abs(model.m_enc - m_co)))
    r_core = model.r[i]
    E_orb = physics.orbital_energy(m_co, model.m_enc[i], r_core)

    # accretion rates at the core (v ~ 0, so v_rel ~ cs)
    mdot_bondi = physics.bhl_accretion_rate(rho_c, m_co, cs_c, 0.0)
    mdot_edd = physics.eddington_accretion_rate(m_co, eta=eta, kappa=kappa)

    # energy from accreting a fraction of the star (efficiency eta)
    E_acc_per_g = eta * cst.C**2
    # time to release U_bind via accretion luminosity at each rate
    t_bondi = U_bind / (eta * mdot_bondi * cst.C**2)
    t_edd = U_bind / (eta * mdot_edd * cst.C**2)
    # mass that must be accreted (at efficiency eta) to supply U_bind
    m_unbind = U_bind / E_acc_per_g

    return dict(
        U_bind=U_bind, E_orb_released=E_orb, r_core=r_core,
        mdot_bondi=mdot_bondi, mdot_edd=mdot_edd,
        t_unbind_bondi=t_bondi, t_unbind_edd=t_edd, m_unbind=m_unbind,
        E_orb_over_Ubind=E_orb / U_bind,
    )
