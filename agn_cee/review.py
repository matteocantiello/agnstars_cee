"""Robustness checks and diagnostics added in response to the first technical review.

Deliberately kept in a separate module (and used only from the appended notebook
section) so the whole review pass can be kept or discarded without touching the
core modules. Addresses:

  R1  sound-speed prescription (gas-only vs full adiabatic vs isothermal-total)
  R3  BBH component-wise validity (gravitational-focusing radius vs binary scale)
  R4  energy deposited into the star by gas hardening vs binding energy
  R5  measurable dephasing after removing the t_c / phi_c degeneracies
  --  total binding energy including internal energy (radiation-dominated star)
  --  MESA mass-column check
  rate disk aspect-ratio self-consistency (tau_mig ~ h^2)

All CGS.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from . import constants as cst
from . import physics
from . import bbh
from . import inspiral
from . import structure
from . import detectors as det
from . import plotstyle

plt = plotstyle.apply()
C = plotstyle.COLORS
COL, DCOL = plotstyle.COL, plotstyle.DCOL
DENS = plotstyle.DENSITY_COLORS
DETC = plotstyle.DETECTOR_COLORS

A_RAD = 4.0 * cst.SIGMA_SB / cst.C        # radiation constant a = 4 sigma/c [erg cm^-3 K^-4]


# --------------------------------------------------------------------------- #
# R1 : sound-speed prescriptions from the MESA interior
# --------------------------------------------------------------------------- #
def _gamma1_mixture(beta):
    """Chandrasekhar (1939) Gamma_1 for an ideal-gas + radiation mixture (gas gamma=5/3).

    beta = P_gas/P_tot. Limits: beta->1 gives 5/3, beta->0 gives 4/3.
    """
    return (32.0 - 24.0 * beta - 3.0 * beta**2) / (24.0 - 21.0 * beta)


def sound_speed_prescriptions(profile_path=structure.DEFAULT_PROFILE):
    """Interior radius and three sound-speed prescriptions from the MESA profile.

    Returns a dict (arrays ordered centre->surface):
      cs_gas  = sqrt(5/3 * P_gas / rho)            (gas-only; the current model)
      cs_tot  = sqrt(Gamma_1 * P_tot / rho)        (full adiabatic, gas+radiation)
      cs_iso  = sqrt(P_tot / rho)                  (isothermal-total; conservative middle)
    plus beta, Gamma_1, and the pressures. P_gas = P_tot - P_rad with P_rad = a T^4/3.
    """
    p = structure.load_mesa_profile(profile_path)
    r = (10.0 ** p.logR)[::-1] * cst.RSUN
    rho = (10.0 ** p.logRho)[::-1]
    T = (10.0 ** p.logT)[::-1]
    P_tot = (10.0 ** p.logP)[::-1]
    P_rad = A_RAD * T**4 / 3.0
    P_gas = np.clip(P_tot - P_rad, 1e-30, None)
    beta = P_gas / P_tot
    g1 = _gamma1_mixture(beta)
    return dict(r=r, rho=rho, T=T, P_tot=P_tot, P_gas=P_gas, P_rad=P_rad,
                beta=beta, gamma1=g1,
                cs_gas=np.sqrt(5.0 / 3.0 * P_gas / rho),
                cs_tot=np.sqrt(g1 * P_tot / rho),
                cs_iso=np.sqrt(P_tot / rho))


def cs_at(pres, r_target, key):
    """Interpolate one prescription ('cs_gas'|'cs_tot'|'cs_iso'|'beta') at radius r [cm]."""
    return float(np.interp(r_target, pres["r"], pres[key]))


# --------------------------------------------------------------------------- #
# R3 : validity of the component-wise BBH drag model
# --------------------------------------------------------------------------- #
def component_focusing(m1, m2, a, cs):
    """Gravitational-focusing (BHL) radius of each BBH component vs its orbital radius.

    For component i (mass m_i) orbiting the c.o.m. at v_i = v_rel m_j/M on radius
    r_i = a m_j/M, returns R_BHL,i / r_i and the Mach number v_i/cs. When this ratio
    is >~ 1 the component's nonlinear gravitational sphere is comparable to the binary
    scale, so the "two independent point perturbers in passive gas" picture is invalid
    *regardless* of the M_gas(<a) > M_BBH criterion. For an equal-mass supersonic binary
    R_BHL/r ~ 8 (constant), i.e. always in the nonlinear/global regime.
    """
    a = np.asarray(a, dtype=float)
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)
    out = {}
    for tag, (mi, mj) in (("1", (m1, m2)), ("2", (m2, m1))):
        vi = v_rel * mj / M
        ri = a * mj / M
        R_bhl = physics.bondi_hoyle_radius(mi, vi, cs)
        out[f"ratio{tag}"] = R_bhl / ri
        out[f"mach{tag}"] = vi / cs
    return out


# --------------------------------------------------------------------------- #
# R4 : energy deposited into the star by gas hardening
# --------------------------------------------------------------------------- #
def gas_energy_deposited(m1, m2, a0, rho, cs, a_end=None, n=6000):
    """Cumulative orbital energy delivered to the gas as the BBH hardens a0 -> a.

    E_gas(a) = integral_a^{a0} L_gas / |da/dt_tot| da'  [erg], i.e. the work done against
    gas drag (which heats / stirs the surrounding stellar material). Returns separation
    grid and E_gas(a) accumulated from a0 inward, plus the internal binding energy of the
    binary E_bin(a) = G m1 m2 / (2a) for comparison.
    """
    if a_end is None:
        a_end = bbh.a_isco(m1, m2)
    a = np.logspace(np.log10(a_end), np.log10(a0), n)
    h = bbh.hardening_rates(m1, m2, a, rho, cs)
    integrand = h["L_gas"] / np.abs(h["dadt_tot"])          # dE_gas/da
    E_cum_in = cumulative_trapezoid(integrand, a, initial=0.0)   # from a_end up to a
    E_gas = E_cum_in[-1] - E_cum_in                          # accumulated from a0 inward
    return dict(a=a, E_gas=E_gas, E_bin=physics.orbital_energy(m1, m2, a),
                f_gw=h["f_gw"])


# --------------------------------------------------------------------------- #
# R5 : measurable dephasing after removing the t_c / phi_c degeneracies
# --------------------------------------------------------------------------- #
def measurable_dephasing(m1, m2, a0, rho, cs, band):
    """Dephasing in a frequency band after maximising over coalescence time & phase.

    The raw delta-N(>f) counts every cycle the gas removes, but a constant phase
    (phi_c) and a term linear in f (t_c) are unmeasurable degeneracies. We take the
    accumulated dephasing phase Psi(f) = 2 pi [N_vac(>f) - N_gas(>f)] over the band,
    subtract its best-fit (a + b f), and report the residual -- a proxy for the
    *measurable* phase distortion (a full result needs an overlap/mismatch integral
    over masses & spins too).

    Returns dict with raw delta-N at the band's low edge and the residual (max & rms,
    in cycles).
    """
    f_lo, f_hi = band
    d = bbh.dephasing(m1, m2, a0, rho, cs)
    f, dN = d["f_gw"], d["dN"]
    m = (f >= f_lo) & (f <= f_hi)
    fb, psi = f[m], 2.0 * np.pi * dN[m]            # dephasing phase [rad]
    if fb.size < 5:
        return dict(raw_dN=np.nan, resid_max=np.nan, resid_rms=np.nan, n=fb.size)
    # remove constant (phi_c) + linear-in-f (t_c)
    A = np.vstack([np.ones_like(fb), fb]).T
    coef, *_ = np.linalg.lstsq(A, psi, rcond=None)
    resid = psi - A @ coef                          # [rad]
    raw = float(dN[int(np.argmin(np.abs(f - f_lo)))])   # dephasing from the band's low edge
    return dict(raw_dN=raw,
                resid_max=float(np.max(np.abs(resid)) / (2 * np.pi)),
                resid_rms=float(np.sqrt(np.mean(resid**2)) / (2 * np.pi)),
                n=int(fb.size))


# --------------------------------------------------------------------------- #
# Total binding energy including internal energy (radiation-dominated star)
# --------------------------------------------------------------------------- #
def binding_energy_components(profile_path=structure.DEFAULT_PROFILE):
    """Gravitational vs internal energy of the immortal, and the *net* binding energy.

    Uses the MESA *mass column* for M(r) (and cross-checks the 4 pi r^2 rho dr
    reconstruction). Internal energy density u = (3/2) P_gas + 3 P_rad (ideal gas +
    radiation). The net energy required to unbind a shell to infinity is reduced by its
    internal energy, so for a radiation-pressure-dominated (near n=3) star the net
    binding energy can be far below |Omega|.
    """
    p = structure.load_mesa_profile(profile_path)
    r = (10.0 ** p.logR)[::-1] * cst.RSUN
    rho = (10.0 ** p.logRho)[::-1]
    T = (10.0 ** p.logT)[::-1]
    P_tot = (10.0 ** p.logP)[::-1]
    M_col = p.mass[::-1] * cst.MSUN                    # MESA mass coordinate
    dm = np.diff(M_col, prepend=0.0)

    # reconstruction check (exclude the central zones where M->0 inflates rel. error)
    dr = np.diff(r)
    dr = np.insert(dr, 0, dr[0])
    M_rec = np.cumsum(4.0 * np.pi * rho * r**2 * dr)
    inner = M_col > 1e-3 * M_col[-1]
    rel_mass_err = float(np.max(np.abs(M_rec[inner] - M_col[inner]) / M_col[inner]))

    # gravitational binding |Omega| = int G M(r)/r dm  (>0)
    omega = float(np.sum(cst.G * M_col / np.maximum(r, 1.0) * dm))
    # internal energy E_int = int (u/rho) dm,  u = 3/2 P_gas + 3 P_rad
    P_rad = A_RAD * T**4 / 3.0
    P_gas = np.clip(P_tot - P_rad, 0.0, None)
    u = 1.5 * P_gas + 3.0 * P_rad
    E_int = float(np.sum(u / rho * dm))
    return dict(omega=omega, E_int=E_int, net_binding=omega - E_int,
                ratio_int_over_grav=E_int / omega, M_total=M_col[-1],
                rel_mass_err=rel_mass_err)


# --------------------------------------------------------------------------- #
# Rate : disk aspect-ratio self-consistency
# --------------------------------------------------------------------------- #
def mass_profile_check(profile_path=structure.DEFAULT_PROFILE):
    """Compare the MESA mass coordinate to the 4 pi r^2 rho dr reconstruction.

    The reconstruction (used by structure.build_model) is ~10-20% high in the inner
    region; the MESA mass column is the accurate M(r). Returns r [cm], both M(r) [g],
    and the radius where each gives M_enc = 10 Msun.
    """
    p = structure.load_mesa_profile(profile_path)
    r = (10.0 ** p.logR)[::-1] * cst.RSUN
    rho = (10.0 ** p.logRho)[::-1]
    M_col = p.mass[::-1] * cst.MSUN
    dr = np.diff(r)
    dr = np.insert(dr, 0, dr[0])
    M_rec = np.cumsum(4.0 * np.pi * rho * r**2 * dr)
    r10_col = r[int(np.argmin(np.abs(M_col - 10 * cst.MSUN)))]
    r10_rec = r[int(np.argmin(np.abs(M_rec - 10 * cst.MSUN)))]
    return dict(r=r, M_col=M_col, M_rec=M_rec, r10_col=r10_col, r10_rec=r10_rec)


def encounter_rate_vs_h(h_values, N_bh=10.0):
    """Encounter rate Gamma = N_bh / tau_mig(h) [Myr^-1] vs disk aspect ratio h.

    tau_mig ~ h^2 (Tanaka+2004), so a factor-30 change in h moves Gamma by ~10^3.
    """
    from . import rates
    h = np.asarray(h_values, dtype=float)
    tau = np.array([rates.migration_time(h=hi) for hi in h])     # s
    return N_bh / (tau / rates.MYR)                              # per Myr


# --------------------------------------------------------------------------- #
# Appendix figure builders (publication style; each returns a Figure)
# --------------------------------------------------------------------------- #
SITES_R = [("core", 0.5), ("mid-env.", 7.0), ("outer env.", 11.0)]


def fig_soundspeed(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """R1: sound-speed prescriptions and their effect on the BBH dephasing."""
    pres = sound_speed_prescriptions()
    x = np.log10(pres["r"] / cst.RSUN)
    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 3.1),
                            gridspec_kw={"width_ratios": [1, 1]})

    # panel a: cs prescriptions + beta
    axs[0].plot(x, pres["cs_gas"] / 1e5, color=C[0], lw=1.7, label=r"gas only ($\frac{5}{3}P_{\rm gas}/\rho$)")
    axs[0].plot(x, pres["cs_iso"] / 1e5, color=C[1], lw=1.7, label=r"isothermal-total ($P_{\rm tot}/\rho$)")
    axs[0].plot(x, pres["cs_tot"] / 1e5, color=C[6], lw=1.7, label=r"full adiabatic ($\Gamma_1 P_{\rm tot}/\rho$)")
    axs[0].plot(x, model.cs_of_r(pres["r"]) / 1e5, color="0.5", lw=1.2, ls="--",
                label="current model")
    axs[0].set_xlabel(r"$\log_{10}\,(r/{\rm R}_\odot)$")
    axs[0].set_ylabel(r"$c_{\rm s}$ [km s$^{-1}$]")
    axs[0].legend(fontsize=6.5, loc="upper right")
    axb = axs[0].twinx()
    axb.plot(x, pres["beta"], color=C[4], lw=1.0, ls=":")
    axb.set_ylabel(r"$\beta=P_{\rm gas}/P_{\rm tot}$", color=C[4])
    axb.tick_params(axis="y", colors=C[4])
    axb.set_ylim(0, 1)

    # panel b: core dephasing under each prescription
    rho_c = model.rho_c
    for key, col, lab in [("cs_gas", C[0], "gas only"),
                          ("cs_iso", C[1], "iso-total"),
                          ("cs_tot", C[6], "adiabatic")]:
        cs = cs_at(pres, 0.5 * cst.RSUN, key)
        d = bbh.dephasing(m1, m2, 3 * cst.RSUN, rho_c, cs)
        axs[1].loglog(d["f_gw"], np.clip(d["dN"], 1e-3, None), color=col, lw=1.7, label=lab)
    axs[1].axhline(1.0, color="k", lw=0.7)
    axs[1].axvspan(1e1, 1e3, color=plotstyle.DETECTOR_COLORS["LVK"], alpha=0.06)
    axs[1].set_xlabel(r"GW frequency $f$ [Hz]")
    axs[1].set_ylabel(r"dephasing $\delta N$ [cycles]")
    axs[1].set_xlim(1e-1, 1e3)
    axs[1].set_ylim(1e-2, 1e6)
    axs[1].legend(fontsize=7, loc="upper right", title="core $c_{\\rm s}$:", title_fontsize=7)
    fig.tight_layout(pad=0.4)
    return fig


def fig_bbh_validity(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """R3: gravitational-focusing radius vs binary scale -- component-wise model validity."""
    pres = sound_speed_prescriptions()
    cs = cs_at(pres, 0.5 * cst.RSUN, "cs_gas")
    rho_c = model.rho_c
    a = np.logspace(np.log10(2e-3 * cst.RSUN), np.log10(3 * cst.RSUN), 400)
    cf = component_focusing(m1, m2, a, cs)
    Mgas_ratio = rho_c * (4.0 / 3.0) * np.pi * a**3 / (m1 + m2)

    fig, ax = plt.subplots(figsize=(COL, 3.0))
    ax.loglog(a / cst.RSUN, cf["ratio1"], color=C[6], lw=1.8,
              label=r"$R_{\rm BHL,i}/r_i$ (focusing)")
    ax.loglog(a / cst.RSUN, Mgas_ratio, color=C[0], lw=1.8,
              label=r"$M_{\rm gas}(<a)/M_{\rm BBH}$")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.text(2.2e-3, 1.3, "model invalid above", fontsize=6.5, color="0.3")
    ax.set_xlabel(r"BBH separation $a$ [R$_\odot$]")
    ax.set_ylabel("dimensionless ratio")
    ax.set_ylim(1e-3, 30)
    ax.legend(fontsize=7, loc="lower right")
    axm = ax.twinx()
    axm.loglog(a / cst.RSUN, cf["mach1"], color="0.55", lw=1.0, ls=":")
    axm.set_ylabel("component Mach", color="0.55")
    axm.tick_params(axis="y", colors="0.55")
    fig.tight_layout(pad=0.4)
    return fig


def fig_energy_deposition(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """R4: cumulative gas-deposited energy vs the star's binding energies."""
    rho_c = model.rho_c
    cs = cs_at(sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    e = gas_energy_deposited(m1, m2, 3 * cst.RSUN, rho_c, cs)
    be = binding_energy_components()

    fig, ax = plt.subplots(figsize=(COL, 3.0))
    ax.loglog(e["a"] / cst.RSUN, e["E_gas"], color=C[6], lw=2.0,
              label=r"$E_{\rm gas}(<a)$ deposited")
    ax.axhline(be["omega"], color=C[3], ls="--", lw=1.2, label=r"$|\Omega|$ (gravitational)")
    ax.axhline(be["net_binding"], color=C[0], ls="-.", lw=1.2,
               label=r"net binding ($|\Omega|-E_{\rm int}$)")
    ax.set_xlabel(r"BBH separation $a$ [R$_\odot$]")
    ax.set_ylabel("energy [erg]")
    ax.set_ylim(1e49, 1e54)
    ax.legend(fontsize=7, loc="lower left")
    # mark where E_gas crosses the net binding energy
    i = int(np.argmin(np.abs(e["E_gas"] - be["net_binding"])))
    a_cross = e["a"][i] / cst.RSUN
    ax.plot(a_cross, e["E_gas"][i], "o", color="k", ms=5, zorder=5)
    ax.annotate(f"$E_{{\\rm gas}}$ unbinds core\n($a\\simeq{a_cross:.0e}$ R$_\\odot$)",
                xy=(a_cross, e["E_gas"][i]), xytext=(a_cross * 0.5, 2.5e51),
                fontsize=6.5, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="0.45"))
    fig.tight_layout(pad=0.4)
    return fig


# =========================================================================== #
# BBH gas-hardening model ladder (response to second-review suggestions)
#
# A ladder of prescriptions rather than one "better" formula. Model 1 (Kim08
# double-wake) is the recommended new fiducial; Models 3-4 are systematic
# brackets. Literature basis is noted per function; where exact published fits
# are not reproduced verbatim, the STRUCTURE and documented magnitude/sign are
# parametrized and the knob is exposed (clearly flagged).
# =========================================================================== #
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


def gas_loss_rate_model(m1, m2, a, rho, cs, model=0):
    """Gas energy-loss rate L_gas of the internal orbit for a given ladder model [erg/s]."""
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
        if model == 0:                                   # independent self-wake (current)
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


def dadt_model(m1, m2, a, rho, cs, model=0):
    """Return (da/dt_total, da/dt_GW) for a ladder model [cm/s]."""
    L_gas = gas_loss_rate_model(m1, m2, a, rho, cs, model)
    L_gw = physics.gw_luminosity(m1, m2, a)
    pref = 2.0 * a**2 / (cst.G * m1 * m2)
    return -pref * (L_gas + L_gw), -pref * L_gw


def merger_time_model(m1, m2, a0, rho, cs, model=0, a_end=None, n=4000):
    """Merger time integrating da/(da/dt_total) from a0 to ISCO for a ladder model [s]."""
    if a_end is None:
        a_end = bbh.a_isco(m1, m2)
    a = np.logspace(np.log10(a_end), np.log10(a0), n)
    dadt_tot, _ = dadt_model(m1, m2, a, rho, cs, model)
    return float(np.trapz(1.0 / np.abs(dadt_tot), a))


def hardening_diagnostics(m1, m2, a, rho, cs):
    """Breakdown diagnostics for the independent-wake picture (equal-mass)."""
    a = np.asarray(a, dtype=float)
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)
    vi = v_rel * m2 / M
    ri = a * m1 / M
    R_bhl = physics.bondi_hoyle_radius(m1, vi, cs)
    R_B_bbh = cst.G * M / cs**2
    return dict(a=a,
                RBHL_over_a=R_bhl / a,
                RBHL_over_ri=R_bhl / ri,
                RBbbh_over_a=R_B_bbh / a,
                Mgas_over_Mbbh=rho * (4.0 / 3.0) * np.pi * a**3 / M)


def fig_hardening_ladder(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """Appendix figure: gas- vs GW-driven hardening rate across the full ladder of
    prescriptions. Single clean panel; the validity diagnostics are shown in
    fig_power_caseB, so they are not repeated here."""
    cs = cs_at(sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    rho_c = model.rho_c
    a = np.logspace(np.log10(3e-3 * cst.RSUN), np.log10(3 * cst.RSUN), 400)

    # short legend labels (the full names in HARDENING_MODELS are too long for a
    # single-column legend; each rung is described in Appendix B)
    short = {0: "independent", 1: "linear + KKSS08", 2: "nonlinear + KKSS08 (fid.)",
             3: "CE wind-tunnel", 4: "shared-Bondi", 5: "scaled companion"}
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    styles = {0: ("0.45", "--"), 1: (C[1], ":"), 2: (C[6], "-"),
              3: (C[0], "-."), 4: (C[4], ":"), 5: (C[3], (0, (3, 1, 1, 1)))}
    for mdl in range(6):
        dadt_tot, dadt_gw = dadt_model(m1, m2, a, rho_c, cs, model=mdl)
        col, ls = styles[mdl]
        lw = 2.2 if mdl == 2 else 1.3
        ax.loglog(a / cst.RSUN, np.abs((dadt_tot - dadt_gw) / dadt_gw),
                  color=col, ls=ls, lw=lw, label=short[mdl])
    ax.axhline(1.0, color="k", lw=0.8)
    ax.text(0.97, 0.045, "gas-dominated hardening", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="0.35")
    ax.set_xlabel(r"BBH separation $a$ [R$_\odot$]")
    ax.set_ylabel(r"hardening-rate ratio $|\dot a_{\rm gas}|/|\dot a_{\rm GW}|$")
    ax.set_xlim(a.min() / cst.RSUN, a.max() / cst.RSUN)
    ax.set_ylim(1e-2, 1e15)
    leg = ax.legend(fontsize=6.5, loc="upper left", labelspacing=0.25, ncol=1,
                    handlelength=2.2, borderaxespad=0.4, frameon=True, framealpha=0.9)
    leg.get_frame().set_edgecolor("none")
    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Case B "power figure" (Fig 2 analog) and characteristic-strain plot
# --------------------------------------------------------------------------- #
def fig_power_caseB(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, d_obs=100 * cst.MPC):
    """Fig-2 analog for Case B: energy-loss channels vs the BBH *internal* separation,
    with the gas-drag model band (fiducial = solid) + GW power + diagnostics."""
    from matplotlib.transforms import blended_transform_factory
    RS = cst.RSUN
    rho_c = model.rho_c
    cs = cs_at(sound_speed_prescriptions(), 0.5 * RS, "cs_gas")
    a = np.logspace(np.log10(bbh.a_isco(m1, m2)), np.log10(3 * RS), 500)
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)

    L_gw = physics.gw_luminosity(m1, m2, a)
    band = np.array([gas_loss_rate_model(m1, m2, a, rho_c, cs, model=mm) for mm in (0, 2, 3, 4, 5)])
    Lg_lo, Lg_hi = np.clip(band.min(0), 1e-30, None), band.max(0)
    Lg_fid = gas_loss_rate_model(m1, m2, a, rho_c, cs, model=2)
    L_acc = np.zeros_like(a)
    for mi, mj in ((m1, m2), (m2, m1)):
        vi = v_rel * mj / M
        L_acc = L_acc + physics.accretion_drag_force(rho_c, mi, cs, vi) * vi
    f_gw, h = physics.gw_strain(m1, m2, a, d_obs)

    fig, axs = plt.subplots(2, 1, figsize=(COL, 4.6), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    plt.subplots_adjust(hspace=0.04)
    ax = axs[0]
    ax.fill_between(a / RS, Lg_lo, Lg_hi, color="0.75", alpha=0.6, lw=0,
                    label="gas hardening (model range)")
    ax.loglog(a / RS, Lg_fid, color=C[1], lw=2.2, label="gas hardening (fiducial)")
    ax.loglog(a / RS, L_gw, color=C[6], lw=2.0, label="GW power (strain)")
    ax.loglog(a / RS, L_acc, color=C[0], lw=1.2, ls="--", label="accretion drag")
    ax.set_ylabel(r"power [erg s$^{-1}$]")
    ax.set_xlim(a.min() / RS, a.max() / RS)
    ax.set_ylim(1e38, 1e56)
    leg = ax.legend(loc="lower left", fontsize=6.0, frameon=True, framealpha=0.85)
    leg.get_frame().set_edgecolor("none")

    C_hl = np.median(h / L_gw ** 0.2)
    ax3 = ax.twinx(); ax3.set_yscale("log")
    ax3.set_ylim(C_hl * 1e38 ** 0.2, C_hl * 1e54 ** 0.2)
    ax3.set_ylabel(r"GW strain $h$ (100 Mpc)", color=C[6])
    ax3.tick_params(axis="y", colors=C[6]); ax3.spines["right"].set_color(C[6])

    f_sel = np.power(10.0, np.round(np.log10(np.logspace(
        np.log10(f_gw.min()), np.log10(f_gw.max()), 5))))
    a_for_f = physics.separation_from_gw_frequency(m1, m2, f_sel) / RS
    ax2 = ax.twiny(); ax2.set_xscale("log"); ax2.set_xlim(a.min() / RS, a.max() / RS)
    ax2.set_xticks(a_for_f); ax2.set_xticklabels([r"$10^{%.0f}$" % np.log10(ff) for ff in f_sel])
    ax2.set_xlabel(r"$f_{\rm GW}$ [Hz]", labelpad=3)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for name, f0, f1 in [("LISA", 1e-4, 1e-1), ("DECIGO", 1e-2, 1e1), ("LVK", 1e1, 1e3)]:
        a_hi = physics.separation_from_gw_frequency(m1, m2, f1) / RS
        a_lo = physics.separation_from_gw_frequency(m1, m2, f0) / RS
        lo, hi = max(a_hi, a.min() / RS), min(a_lo, a.max() / RS)
        if hi > lo:
            ax.axvspan(lo, hi, color=DETC[name], alpha=0.06)
            ax.text(np.sqrt(lo * hi), 0.95, name, transform=trans, ha="center",
                    va="top", fontsize=7.5, color=DETC[name], fontweight="bold")

    dg = hardening_diagnostics(m1, m2, a, rho_c, cs)
    axb = axs[1]
    axb.loglog(a / RS, dg["RBHL_over_ri"], color=C[6], lw=1.6, label=r"$R_{\rm BHL,i}/r_i$")
    axb.loglog(a / RS, dg["RBbbh_over_a"], color=C[0], lw=1.6, label=r"$R_{\rm B,BBH}/a$")
    axb.loglog(a / RS, dg["Mgas_over_Mbbh"], color=C[1], lw=1.6, label=r"$M_{\rm gas}/M_{\rm BBH}$")
    axb.axhline(1.0, color="k", lw=0.8)
    axb.set_xlabel(r"BBH separation $a$ [R$_\odot$]"); axb.set_ylabel("validity diag.")
    axb.set_ylim(1e-3, 1e3); axb.legend(fontsize=5.5, loc="lower left")
    fig.tight_layout(pad=0.4)
    return fig


def fig_characteristic_strain(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, distance=100 * cst.MPC):
    """Characteristic strain h_c(f) of the gas-hardened BBH on the LISA/DECIGO/LVK curves.

    h_c = A(f) sqrt(n), n = f^2/|fdot| (cycles near f); gas raises |fdot| -> fewer cycles ->
    suppressed h_c in the gas-dominated (LISA) band, recovering to ~vacuum near merger (LVK).
    """
    rho_c = model.rho_c
    cs = cs_at(sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    a = np.logspace(np.log10(bbh.a_isco(m1, m2)), np.log10(5 * cst.RSUN), 800)
    f, A = physics.gw_strain(m1, m2, a, distance)
    dfda = np.gradient(f, a)
    T_obs = 4.0 * 365.25 * 86400.0   # nominal mission/observation time [s]

    def hc_of(dadt):
        # h_c = A sqrt(n_cyc), with the effective cycle count capped at f*T_obs:
        # n_cyc = min(f^2/|fdot|, f*T_obs). For a slowly-evolving (e.g. vacuum, mHz)
        # source the residence time f^2/|fdot| exceeds the mission, so the finite
        # observation time T_obs limits the accumulated cycles (and hence h_c).
        fdot = np.abs(dfda * dadt)
        n_resid = f**2 / np.clip(fdot, 1e-300, None)
        n_cyc = np.minimum(n_resid, f * T_obs)
        return A * np.sqrt(np.clip(n_cyc, 0.0, None))

    hc_vac = hc_of(dadt_model(m1, m2, a, rho_c, cs, model=2)[1])
    band = np.array([hc_of(dadt_model(m1, m2, a, rho_c, cs, model=mm)[0]) for mm in (0, 2, 3, 4, 5)])
    hc_lo, hc_hi = band.min(0), band.max(0)
    hc_fid = hc_of(dadt_model(m1, m2, a, rho_c, cs, model=2)[0])

    fig, ax = plt.subplots(figsize=(DCOL, 4.0))
    fg = np.logspace(-4, 4, 2000)
    for name in ("LISA", "DECIGO", "LVK"):
        # sky-averaged sensitivity (LVK raw ASD scaled by 1/SKY_AVG=2.5; LISA/DECIGO already averaged)
        hn = np.sqrt(fg * det.DETECTORS[name](fg)) / det.SKY_AVG[name]
        ax.loglog(fg, np.where(np.isfinite(hn), hn, np.nan), color=DETC[name], lw=1.4, label=name)
    ax.fill_between(f, hc_lo, hc_hi, color="0.75", alpha=0.6, lw=0)
    ax.loglog(f, hc_vac, color="0.4", lw=1.3, ls="--", label="same BBH, vacuum")
    ax.loglog(f, hc_fid, color=C[1], lw=2.1, label="BBH in immortal (fiducial)")
    ax.set_xlim(1e-4, 1e4); ax.set_ylim(1e-26, 2*1e-18)
    ax.set_xlabel("frequency [Hz]"); ax.set_ylabel(r"characteristic strain $h_c$")
    leg = ax.legend(fontsize=10, loc="lower right", ncol=2, frameon=True, framealpha=0.85)
    leg.get_frame().set_edgecolor("none")
    ax.text(0.7, 0.91, f"10+10 M$_\\odot$ at {distance/cst.MPC:.0f} Mpc",
            transform=ax.transAxes, fontsize=12)
    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Queued Case B plots: a(t) trajectory and eccentricity-regime diagnostic
# --------------------------------------------------------------------------- #
def fig_caseB_trajectory(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, a0=cst.RSUN):
    """a(t) of the BBH internal orbit at the core: the plunge from a0 to merger
    (fiducial Model 2 solid, ladder range shaded) -- the Case B analog of Fig 3."""
    rho_c = model.rho_c
    cs = cs_at(sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    a = np.logspace(np.log10(bbh.a_isco(m1, m2)), np.log10(a0), 4000)

    def t_since(mdl):
        dadt, _ = dadt_model(m1, m2, a, rho_c, cs, model=mdl)
        t_in = cumulative_trapezoid(1.0 / np.abs(dadt), a, initial=0.0)  # a -> merger
        return (t_in[-1] - t_in) / 3600.0                               # hr since capture at a0

    t_fast, t_slow, t_fid = t_since(0), t_since(4), t_since(2)
    fig, ax = plt.subplots(figsize=(COL, 3.2))
    ax.fill_betweenx(a / cst.RSUN, t_fast, t_slow, color="0.8", alpha=0.7, lw=0,
                     label="ladder range")
    ax.plot(t_fid, a / cst.RSUN, color=C[6], lw=2.2, label="fiducial (Model 2)")
    ax.set_yscale("log")
    ax.set_xlabel("time since BBH capture [hr]")
    ax.set_ylabel(r"BBH separation $a$ [R$_\odot$]")
    ax.set_xlim(0, t_slow.max() * 1.02)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.1, 0.9, r"10+10 M$_\odot$, core", transform=ax.transAxes, fontsize=10)
    fig.tight_layout(pad=0.4)
    return fig


def fig_eccentricity_regime(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """Where gas circularizes vs excites the internal eccentricity (regime map).

    O'Neill+2024: gaseous DF circularizes a *subsonic* orbit and can *excite* e when
    *supersonic*; GW always circularizes (Peters) and dominates below the gas=GW crossover.
    So e can build only in the supersonic + gas-dominated window -- which sits in the deci-Hz
    band -- and is then damped by GW near merger. (KKSS08 fits are circular-orbit, so this is a
    regime diagnostic, not an e(a) integration.)"""
    rho_c = model.rho_c
    cs = cs_at(sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    a = np.logspace(np.log10(bbh.a_isco(m1, m2)), np.log10(3 * cst.RSUN), 600)
    M = m1 + m2
    v_rel = physics.v_kepler(m1, m2, a)
    mach = (v_rel * m2 / M) / cs
    f = physics.gw_frequency(m1, m2, a)
    h = bbh.hardening_rates(m1, m2, a, rho_c, cs)
    ratio = np.abs(h["dadt_gas"] / h["dadt_gw"])

    # boundaries: Mach=1 (sonic) and gas=GW crossover
    f_sonic = float(f[np.argmin(np.abs(mach - 1.0))])
    f_cross = float(f[np.argmin(np.abs(ratio - 1.0))])

    fig, ax = plt.subplots(figsize=(DCOL, 3.6))
    ax.axvspan(f.min(), f_sonic, color=C[1], alpha=0.10)
    ax.axvspan(f_sonic, f_cross, color=C[4], alpha=0.13)
    ax.axvspan(f_cross, f.max(), color=C[6], alpha=0.08)
    ax.loglog(f, mach, color="k", lw=2.0)
    ax.axhline(1.0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"GW frequency $f$ [Hz]")
    ax.set_ylabel("component Mach  $v_i/c_s$")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(0.04, 60)

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    # region labels along the bottom (clear of the Mach curve and the top detector bars)
    ax.text(np.sqrt(f.min() * f_sonic), 0.06, "subsonic\n(circularize)", transform=trans,
            ha="center", va="bottom", fontsize=9, color=C[1])
    ax.text(np.sqrt(f_sonic * f_cross), 0.06,
            "supersonic, gas-dominated\n$e$ grows (O'Neill+24): deci-Hz", transform=trans,
            ha="center", va="bottom", fontsize=9, color=plotstyle.VERMILLION)
    ax.text(np.sqrt(f_cross * f.max()), 0.06, "GW-dominated\n(circularize)", transform=trans,
            ha="center", va="bottom", fontsize=9, color=C[6])
    # detector bars + labels along the top
    for name, f0, f1 in [("LISA", 1e-4, 1e-1), ("DECIGO", 1e-2, 1e1), ("LVK", 1e1, 1e3)]:
        lo, hi = max(f0, f.min()), min(f1, f.max())
        if hi > lo:
            ax.plot([lo, hi], [0.965, 0.965], transform=trans, color=DETC[name], lw=4,
                    solid_capstyle="butt", clip_on=False)
            ax.text(np.sqrt(lo * hi), 0.905, name, transform=trans, ha="center", va="top",
                    fontsize=9, color=DETC[name], fontweight="bold")
    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Spiral-in trajectories (Case A single CO; Case B internal BBH) + breakdown
# --------------------------------------------------------------------------- #
def fig_true_trajectories(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    """Real (integrated) spiral-in trajectories. Both panels plot the genuine path
    (r cos phi, r sin phi) with the true accumulated orbital phase
    phi(t) = int Omega dt; nothing is rescaled.

    (a) A single compact object sinking from the stellar surface to the core
    (linear radius, true geometry): it orbits the enclosed stellar mass
    M_enc(r), lingers near the tenuous surface, then plunges through the dense
    interior. The pink disk marks r < r(M_star = m_BH), where the test-particle
    treatment fails.
    (b) The same with a second black hole (BH1) held at the centre, followed to
    merger (log radius). BH2 orbits M_enc(r) + m1 at the local gas density --
    identical drag physics to (a) -- and the GW comes from the BH2-BH1 pair.
    The binary winds ~3000 times, crowding into the gas-GW stall near a few
    1e-3 Rsun, then plunges to the ISCO. The pink disk marks R_BHL/r > 1
    (r < ~0.6 Rsun), the 3D-breakdown region. Colour is the time remaining to the
    endpoint (merger in b, the core in a), log-scaled and floored at 1 s, so the
    timescale collapse from ~0.1 yr at the surface to seconds at merger is visible.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm
    G, Rsun, GRAY = cst.G, cst.RSUN, "0.9"
    PINK, DPINK, PINK_BG, CMAP = "#ff5c8a", "#b3004d", "#ffe1ea", "viridis_r"

    def colored_line(ax, x, y, cval, lw, norm):
        pts = np.array([x, y]).T.reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(seg, cmap=CMAP, lw=lw, zorder=3, rasterized=True, norm=norm)
        lc.set_array(np.clip(cval, norm.vmin, norm.vmax)); ax.add_collection(lc)
        return lc

    # ---- (a) single CO: true integrated spiral, surface -> core (linear radius) ----
    solA = inspiral.spiral_in_time(model, m2, model.r_shock, n=200000)
    rA, mencA, LtA = solA["r"], solA["m_enc"], solA["L_total"]
    omA = np.sqrt(G * mencA / rA**3)
    LangA = m2 * np.sqrt(G * mencA * rA)
    dtA = np.gradient(LangA, rA) * omA / LtA
    phiA = cumulative_trapezoid(omA * dtA, rA, initial=0.0)
    tA = cumulative_trapezoid(dtA, rA, initial=0.0)
    phA = np.linspace(0.0, phiA[-1], int(phiA[-1] / (2 * np.pi) * 200))
    rrA = np.interp(phA, phiA, rA) / Rsun
    tmergeA = np.interp(phA, phiA, tA)                    # time remaining to the core [s]
    xA, yA = rrA * np.cos(phA), rrA * np.sin(phA)
    Rstar, rbreak = model.r_shock / Rsun, solA["r_final"] / Rsun

    # ---- (b) CO + central BH: true trajectory all the way to the ISCO (log radius) ----
    # channel_luminosities has no inner floor (spiral_in_time clips at the profile
    # edge), so we integrate the binary hardening into the uniform core (rho->rho_c).
    aend, asurf = bbh.a_isco(m1, m2), model.r_shock
    rB = np.logspace(np.log10(aend), np.log10(asurf), 300000)
    chB = inspiral.channel_luminosities(model, m2, rB, m_central=m1)
    vcB, mencB, csB = chB["v_c"], chB["m_enc"], chB["cs"]
    omB = vcB / rB
    LangB = m2 * np.sqrt(G * mencB * rB)
    dtB = np.gradient(LangB, rB) * omB / chB["L_total"]
    phiB = cumulative_trapezoid(omB * dtB, rB, initial=0.0)
    tB = cumulative_trapezoid(dtB, rB, initial=0.0)
    RBHL_r = (2 * G * m2 / (vcB**2 + csB**2)) / rB        # per-component Bondi / orbital radius
    ib = np.where(RBHL_r > 1)[0]
    a_brk = rB[ib[-1]] if len(ib) else rB[0]
    umax = np.log10(asurf / aend)
    _rp = lambda av: np.log10(av / aend) / umax          # 0 = ISCO (centre), 1 = surface
    phB = np.linspace(0.0, phiB[-1], int(phiB[-1] / (2 * np.pi) * 50))
    rpB = _rp(np.interp(phB, phiB, rB))
    tmergeB = np.interp(phB, phiB, tB)                    # time remaining to merger [s]
    xB, yB = rpB * np.cos(phB), rpB * np.sin(phB)

    # log colour: time remaining to the endpoint, floored at 1 s (spans ~0.1 yr -> seconds)
    norm = LogNorm(vmin=1.0, vmax=max(tmergeA.max(), tmergeB.max()))
    FRAC = 0.88
    La, Lb = Rstar / FRAC, 1.0 / FRAC                    # same circle-to-box fraction in both

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(DCOL, 3.95))
    fig.subplots_adjust(left=0.065, right=0.875, bottom=0.13, top=0.97, wspace=0.24)

    # (a) single compact object -- linear radius, true geometry
    axA.add_patch(plt.Circle((0, 0), Rstar, fc=GRAY, ec="k", lw=1.4, zorder=1))
    axA.add_patch(plt.Circle((0, 0), rbreak, fc=PINK, ec="none", alpha=0.9, zorder=2))
    colored_line(axA, xA, yA, tmergeA, 0.45, norm)
    axA.plot(0, 0, "+", color="k", ms=8, mew=1.4, zorder=5)
    axA.set_xlim(-La, La); axA.set_ylim(-La, La); axA.set_aspect("equal")
    axA.set_xlabel(r"$x$ [$R_\odot$]"); axA.set_ylabel(r"$y$ [$R_\odot$]")
    axA.text(0.035, 0.965, "(a)", transform=axA.transAxes, va="top", ha="left",
             fontweight="bold", fontsize=11)

    # (b) compact object + central BH -- log radius, to the ISCO
    axB.add_patch(plt.Circle((0, 0), 1.0, fc=GRAY, ec="none", zorder=0))             # stellar interior
    axB.add_patch(plt.Circle((0, 0), _rp(a_brk), fc=PINK_BG, ec="none", zorder=1))
    lcB = colored_line(axB, xB, yB, tmergeB, 0.2, norm)
    axB.add_patch(plt.Circle((0, 0), 1.0, fc="none", ec="k", lw=1.4, zorder=4))   # stellar surface
    axB.plot(0, 0, "+", color="k", ms=8, mew=1.4, zorder=6)
    axB.text(0.5, 0.785, r"needs 3D ($R_{\rm BHL}/r>1$)", transform=axB.transAxes,
             color=DPINK, ha="center", va="center", fontsize=7.0, zorder=6,
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    tk = [_rp(av) for av in (1e-3 * Rsun, 1e-2 * Rsun, 0.1 * Rsun, Rsun, 10 * Rsun)]
    tl = [r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$1$", r"$10$"]
    axB.set_xticks([-t for t in tk[::-1]] + tk); axB.set_xticklabels(tl[::-1] + tl, fontsize=6.0)
    axB.set_yticks([-t for t in tk[::-1]] + tk); axB.set_yticklabels(tl[::-1] + tl, fontsize=6.0)
    axB.set_xlim(-Lb, Lb); axB.set_ylim(-Lb, Lb); axB.set_aspect("equal")
    axB.set_xlabel(r"$r$ [$R_\odot$] (log radius)"); axB.set_ylabel(r"$r$ [$R_\odot$] (log radius)")
    axB.text(0.035, 0.965, "(b)", transform=axB.transAxes, va="top", ha="left",
             fontweight="bold", fontsize=11)

    # colorbar pinned to exactly the height of the (square, equal-aspect) panels
    fig.canvas.draw()
    pos = axB.get_position()
    cax = fig.add_axes([pos.x1 + 0.013, pos.y0, 0.015, pos.height])
    cb = fig.colorbar(lcB, cax=cax)
    cb.set_label("time to merger / core [s]")
    return fig
