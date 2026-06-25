"""Figure builders for the paper / notebook.

Each ``fig_*`` returns a publication-styled matplotlib Figure (no on-figure
titles -- descriptions belong in the LaTeX captions). The physics lives in the
other ``agn_cee`` modules; here we only assemble the figures. Sizes follow the
AAS template: ``COL`` single-column, ``DCOL`` double-column.
"""

import numpy as np

from . import constants as cst
from . import physics
from . import inspiral
from . import bbh
from . import detectors as det
from . import observability as obs
from . import plotstyle

plt = plotstyle.apply()
C = plotstyle.COLORS
COL, DCOL = plotstyle.COL, plotstyle.DCOL
DENS = plotstyle.DENSITY_COLORS                 # core, mid, outer
DETC = plotstyle.DETECTOR_COLORS                # LISA, DECIGO, LVK

# Representative BBH locations in the star: (label, radius / Rsun).
BBH_SITES = [("core ($r=0.5\\,$R$_\\odot$)", 0.5),
             ("mid-env. ($r=7\\,$R$_\\odot$)", 7.0),
             ("outer env. ($r=11\\,$R$_\\odot$)", 11.0)]


def site_conditions(model):
    out = []
    for label, rr in BBH_SITES:
        r = rr * cst.RSUN
        out.append((label, float(model.rho_of_r(r)), float(model.cs_of_r(r))))
    return out


# --------------------------------------------------------------------------- #
# Figure 1 : structure
# --------------------------------------------------------------------------- #
def fig_structure(model):
    x = np.log10(model.r / cst.RSUN)
    xshock = np.log10(model.r_shock / cst.RSUN)

    fig, axs = plt.subplots(2, 2, figsize=(DCOL, 5.0), sharex="col")

    axs[0, 0].plot(x, np.log10(model.rho), color=C[6])
    axs[0, 0].set_ylabel(r"$\log_{10}\,\rho\ \mathrm{[g\,cm^{-3}]}$")
    axs[0, 0].text(3.4, -11, r"$\rho \propto r^{-3/2}$", fontsize=8)
    axs[0, 0].text(2.4, 2.6, "analytic", fontsize=8)
    axs[0, 0].text(-1.3, 2.6, "MESA", fontsize=8)
    axs[0, 0].text(1.35, -15, "shock", fontsize=8, rotation=90, va="bottom")

    axs[0, 1].plot(x, np.log10(model.cs / 1e5), color=C[6])
    axs[0, 1].set_ylabel(r"$\log_{10}\,c_{\rm s}\ \mathrm{[km\,s^{-1}]}$")

    axs[1, 0].plot(x, np.log10(model.m_enc / cst.MSUN), color=C[6])
    axs[1, 0].set_xlabel(r"$\log_{10}\,(r/{\rm R}_\odot)$")
    axs[1, 0].set_ylabel(r"$\log_{10}\,(M_{\rm enc}/{\rm M}_\odot)$")
    axs[1, 0].text(2.2, -3.4, "accretion stream", fontsize=8)
    axs[1, 0].text(-1.3, -3.4, "H.E. star", fontsize=8)

    with np.errstate(divide="ignore"):
        logv = np.log10(model.v_stream / 1e5)
    axs[1, 1].plot(x, logv, color=C[6])
    axs[1, 1].set_xlabel(r"$\log_{10}\,(r/{\rm R}_\odot)$")
    axs[1, 1].set_ylabel(r"$\log_{10}\,v_{\rm stream}\ \mathrm{[km\,s^{-1}]}$")
    axs[1, 1].text(3.2, 2.7, r"$v \propto r^{-1/2}$", fontsize=8)

    for ax in axs.flat:
        ax.axvline(x=xshock, color=plotstyle.ORANGE, linestyle=":", lw=1.0)

    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 : competing powers vs separation
# --------------------------------------------------------------------------- #
def fig_power(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, d_obs=100 * cst.MPC):
    from matplotlib.transforms import blended_transform_factory
    RS = cst.RSUN
    a = np.linspace(1e6, 1e13, 1000)
    rho, cs = model.rho_of_r(a), model.cs_of_r(a)
    # BH2's orbital speed at separation a is set by the total interior mass: the
    # BH binary (m1+m2) plus the enclosed stellar mass M_star(<a), with BH1 held
    # at the stellar center. Reduces to the binary speed in the deep core
    # (M_enc -> 0) and to sqrt(G M_star/a) out in the envelope.
    menc = np.clip(model.menc_of_r(a), 0.0, None)   # guard tiny-a extrapolation
    vk = np.sqrt(physics.v_kepler(m1, m2, a) ** 2
                 + physics.v_circular(menc, a) ** 2)

    rmin = physics.canto_rmin(m2, vk)
    P_df = physics.dynamical_friction_luminosity(m2, vk, a, cs, rho, rmin)
    P_acc = physics.accretion_drag_luminosity(rho, m2, cs, vk)
    P_gw = physics.gw_luminosity(m1, m2, a)
    eta = 0.01
    L_acc = eta * physics.bhl_accretion_rate(rho, m2, cs, vk) * cst.C**2
    L_edd = physics.eddington_luminosity(m2)
    EB = np.abs(model.U_bind)
    f_gw, h = physics.gw_strain(m1, m2, a, d_obs)
    mach = vk / cs
    r_eq = model.r[int(np.argmin(np.abs(model.m_enc - m2)))]
    r_merger = physics.schwarzschild_radius(m1) + physics.schwarzschild_radius(m2)

    fig, axs = plt.subplots(2, 1, figsize=(DCOL, 6.0), sharex=True,
                            gridspec_kw={"height_ratios": [2, 1]})
    plt.subplots_adjust(hspace=0.04)

    # binding energy: extend flat to the left edge (negligible mass interior to
    # the innermost zone, so U(>r) -> total there).
    r_eb = np.concatenate(([r_merger], model.r))
    EB_eb = np.concatenate(([EB[0]], EB))

    ax = axs[0]
    ax.loglog(a / RS, P_df, color=C[1], lw=2.0, label="dynamical friction (single perturber)")
    ax.loglog(a / RS, P_acc, color=C[0], lw=2.0, label="accretion drag")
    ax.loglog(a / RS, P_gw, color=C[6], lw=2.0, label="GW power (strain)")
    ax.loglog(r_eb / RS, EB_eb, color=C[3], ls=":", lw=1.8, label="stellar binding energy")
    ax.loglog(a / RS, L_acc, color=C[7], ls="--", lw=1.4,
              label=r"BHL accretion luminosity ($\eta{=}0.01$)")
    ax.axhline(L_edd, color="k", ls=":", lw=1.1, label="Eddington luminosity")
    ax.axvline(r_eq / RS, color="k", ls="-.", lw=1.1, label=r"$M_\star(<r){=}M_2$")
    ax.set_ylabel(r"power [erg s$^{-1}$]")
    ax.set_xlim(r_merger / RS, a.max() / RS)
    ax.set_ylim(1e27, 1e64)            # extra headroom so the band labels clear the top axis
    leg = ax.legend(loc="lower left", fontsize=8, labelspacing=0.3,
                    # title=r"$10+10\,$M$_\odot$ at $100$ Mpc",
                    title_fontsize=8, frameon=True, framealpha=1.00)
    leg.get_frame().set_edgecolor("none")

    ax.text(0.8,0.88,r"$10+10\,$M$_\odot$",transform=ax.transAxes,fontsize=12) 

    # detector frequency bands, mapped onto the separation axis (same style as Fig. 7)
    detband = [("LISA", 1e-4, 1e-1), ("DECIGO", 1e-2, 1e1), ("LVK", 1e1, 1e3)]
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j, (name, f0, f1) in enumerate(detband):
        a_hi = physics.separation_from_gw_frequency(m1, m2, f1) / RS  # high f -> small a
        a_lo = physics.separation_from_gw_frequency(m1, m2, f0) / RS
        ax.axvspan(a_hi, a_lo, color=DETC[name], alpha=0.06)
        y = 0.83 + 0.04 * j               # LISA lowest (low-curve side), LVK highest
        ax.plot([a_hi, a_lo], [y, y], transform=trans, color=DETC[name], lw=5.0,
                solid_capstyle="butt", clip_on=False)
        ax.text(np.sqrt(a_hi * a_lo), y + 0.008, name, transform=trans, ha="center",
                va="bottom", fontsize=7.5, color=DETC[name], fontweight="bold")

    # GW strain h and GW power L_GW are both exact power laws of separation a
    # (h ~ a^-1, L_GW ~ a^-5), so h = C * L_GW^(1/5). Rescaling the right axis by
    # this relation (one strain decade per five power decades) lets the single
    # blue GW-power curve also be read as GW strain -- no separate strain line.
    C_hl = np.median(h / P_gw ** 0.2)
    P_LO, P_HI = 1e27, 1e64
    ax3 = ax.twinx()
    ax3.set_yscale("log")
    ax3.set_ylim(C_hl * P_LO ** 0.2, C_hl * P_HI ** 0.2)
    ax3.set_ylabel(r"GW strain $h$ (100 Mpc)", color=C[6])
    ax3.tick_params(axis="y", colors=C[6])
    ax3.spines["right"].set_color(C[6])

    f_sel = np.power(10.0, np.round(np.log10(
        np.logspace(np.log10(f_gw.min()), np.log10(f_gw.max()), 5))))
    a_for_f = physics.separation_from_gw_frequency(m1, m2, f_sel) / RS
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(r_merger / RS, a.max() / RS)
    ax2.set_xticks(a_for_f)
    ax2.set_xticklabels([r"$10^{%.0f}$" % np.log10(f) for f in f_sel])
    ax2.set_xlabel(r"$f_{\rm GW}$ [Hz]", labelpad=3)

    axb = axs[1]
    axb.loglog(a / RS, mach, color=C[3], lw=1.8, label=r"$v_{\rm k}/c_{\rm s}$")
    axb1 = axb.twinx()
    axb1.loglog(a / RS, vk / cst.C, color=C[1], lw=1.8, label=r"$v_{\rm k}/c$")
    axb.set_xlabel(r"separation $a$ [R$_\odot$]")
    axb.set_ylabel("Mach")
    axb1.set_ylabel(r"$v_{\rm k}/c$")
    axb.set_xlim(r_merger / RS, a.max() / RS)
    axb.legend(loc="lower left", fontsize=8)
    axb1.legend(loc="upper right", fontsize=8)

    
    
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 : single-CO spiral-in (Case A)
# --------------------------------------------------------------------------- #
def fig_inspiral(model, masses=(5, 10, 30)):
    fig, axs = plt.subplots(2, 1, figsize=(COL, 4.4))
    cols = DENS

    for mco, col in zip(masses, cols):
        sol = inspiral.spiral_in_time(model, mco * cst.MSUN, r0=model.r_shock)
        t_grid, r_of_t = inspiral.trajectory(sol)
        axs[0].plot(t_grid / 86400.0, r_of_t / cst.RSUN, color=col, lw=1.6,
                    label=fr"${mco}\,M_\odot$")
        axs[1].loglog(sol["r"] / cst.RSUN, sol["t_to_r0"] / cst.YR, color=col, lw=1.6,
                      label=fr"${mco}\,M_\odot$")

    axs[0].set_xlabel("time since capture [days]")
    axs[0].set_ylabel(r"orbital radius  $r$ [R$_\odot$]")
    #axs[0].legend(loc="center right", title=r"$m_{\rm CO}$", title_fontsize=8)
    axs[0].axhline(model.r_shock / cst.RSUN, color="0.5", ls="--", lw=0.9)
    #axs[0].text(25, model.r_shock / cst.RSUN * 0.95, "stellar surface",
    #            fontsize=7, color="0.45", ha="center", va="top")

    axs[1].set_xlabel(r"capture radius  $r_0$ [R$_\odot$]")
    axs[1].set_ylabel("spiral-in time [yr]")
    axs[1].legend(loc="upper left", title=r"$m_{\rm CO}$", title_fontsize=9)
    axs[1].set_xlim(0.7, model.r_shock / cst.RSUN * 1.05)

    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 : merger acceleration by gas drag (Case B)
# --------------------------------------------------------------------------- #
def fig_bbh(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN):
    sites = site_conditions(model)
    a0_grid = np.logspace(np.log10(0.02 * cst.RSUN), np.log10(3 * cst.RSUN), 24)

    fig, axs = plt.subplots(2, 1, figsize=(COL, 4.6))

    t_vac = np.array([physics.t_merge_gw(m1, m2, a0) for a0 in a0_grid])
    axs[0].loglog(a0_grid / cst.RSUN, t_vac / cst.YR, "k--", lw=1.4, label="vacuum (Peters)")
    for (label, rho, cs), col in zip(sites, DENS):
        t_gas = np.array([bbh.merger_time(m1, m2, a0, rho, cs)["t_gas"] for a0 in a0_grid])
        axs[0].loglog(a0_grid / cst.RSUN, t_gas / cst.YR, color=col, lw=1.7, label=label)
        a = np.logspace(np.log10(0.003 * cst.RSUN), np.log10(3 * cst.RSUN), 400)
        hh = bbh.hardening_rates(m1, m2, a, rho, cs)
        axs[1].loglog(a / cst.RSUN, np.abs(hh["dadt_gas"] / hh["dadt_gw"]),
                      color=col, lw=1.7, label=label)

    axs[0].set_xlabel(r"initial separation  $a_0$ [R$_\odot$]")
    axs[0].set_ylabel("merger time [yr]")
    axs[0].legend(loc="upper left", fontsize=7)

    axs[1].axhline(1.0, color="0.5", lw=0.9)
    axs[1].text(0.5, 2.5, "gas = GW", fontsize=7, color="0.4")
    axs[1].set_xlabel(r"separation  $a$ [R$_\odot$]")
    axs[1].set_ylabel(r"$|\dot a_{\rm gas}| / |\dot a_{\rm GW}|$")

    fig.tight_layout(pad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Figure 5 : GW dephasing (Case B)
# --------------------------------------------------------------------------- #
def add_dN_chirp_inset(ax, rect=(0.025, 0.08, 0.40, 0.40)):
    """Schematic inset explaining the dephasing diagnostic delta N(>f).

    Starting from the same observed frequency f, the vacuum binary completes more cycles
    before merger (N_vac) than the gas-hardened one, which chirps faster and merges sooner
    with fewer cycles (N_star). Their difference is delta N = N_vac - N_star (the quantity
    on the main vertical axis). NOTE: this is a cycle-COUNT difference, not the area between
    the two chirps -- the gas binary does more cycles early (faster chirp) but merges sooner.
    """
    VAC, GAS, GHOST, RED = "0.4", "#2166ac", "0.78", "#b3001b"
    iax = ax.inset_axes(rect)
    iax.set_facecolor("white"); iax.patch.set_alpha(0.97)
    for s in iax.spines.values():
        s.set_edgecolor("0.5"); s.set_linewidth(0.8)

    f0, amp, bv, bg = 4.0, 0.85, 1.4, -1.4
    tv = np.linspace(0, 1, 1600)
    yv = amp * np.sin(2 * np.pi * (f0 * tv + 0.5 * 8.0 * tv**2))
    tmg = 0.6
    tg = np.linspace(0, tmg, 1000)
    yg = amp * np.sin(2 * np.pi * (f0 * tg + 0.5 * 14.4 * tg**2))
    tgh = np.linspace(tmg, 1, 700)
    ygh = amp * np.sin(2 * np.pi * (f0 * tgh + 0.5 * 14.4 * tgh**2))

    iax.plot(tgh, ygh + bg, color=GHOST, lw=1.0, ls=(0, (2, 1.6)))   # cycles never completed
    iax.plot(tv, yv + bv, color=VAC, lw=1.4)
    iax.plot(tg, yg + bg, color=GAS, lw=1.8)
    iax.plot([1.0, 1.0], [bv - amp - 0.15, bv + amp + 0.15], color=VAC, lw=0.8, ls="--")
    iax.plot([tmg, tmg], [bg - amp - 0.15, bg + amp + 0.15], color=GAS, lw=0.8, ls="--")
    iax.plot(1.0, bv, "o", color=VAC, ms=4.5, zorder=6)
    iax.plot(tmg, bg, "o", color=GAS, ms=5.0, zorder=6)

    # all labels live in the top / centre / bottom margins, so nothing overlaps the chirps
    top, bot = bv + amp + 0.12, bg - amp - 0.12
    iax.text(0.02, top, "vacuum", color=VAC, fontsize=6.8, fontweight="bold", ha="left", va="bottom")
    iax.text(0.99, top, r"$N_{\rm vac}$ cycles", color=VAC, fontsize=6.6, ha="right", va="bottom")
    iax.text(0.02, bot, "in star: merges sooner", color=GAS, fontsize=6.6, fontweight="bold",
             ha="left", va="top")
    iax.text(0.99, bot, r"$N_{\rm star}$ cycles", color=GAS, fontsize=6.6, ha="right", va="top")
    iax.text(0.5, 0.0, r"$\delta N = N_{\rm vac}-N_{\rm star}$", fontsize=8.2, color=RED,
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=RED, lw=0.9))
    iax.set_xlim(-0.02, 1.02); iax.set_ylim(bg - amp - 0.7, bv + amp + 0.7)
    iax.set_xticks([]); iax.set_yticks([])
    return iax


def fig_dephasing(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, a0=5 * cst.RSUN):
    from matplotlib.transforms import blended_transform_factory
    from matplotlib.lines import Line2D
    sites = site_conditions(model)
    detband = [("LISA", 1e-4, 1e-1), ("DECIGO", 1e-2, 1e1), ("LVK", 1e1, 1e3)]

    fig, ax = plt.subplots(figsize=(DCOL, 4.2))
    for name, f0, f1 in detband:
        ax.axvspan(f0, f1, color=DETC[name], alpha=0.06)

    d0 = bbh.dephasing(m1, m2, a0, sites[0][1], sites[0][2])
    ax.loglog(d0["f_gw"], np.clip(d0["N_vac"], 1e-12, None), color="0.55",
              lw=1.1, ls="--", label=r"vacuum cycles $N_{\rm vac}(>f)$")

    for (label, rho, cs), col in zip(sites, DENS):
        d = bbh.dephasing(m1, m2, a0, rho, cs)
        f, dN = d["f_gw"], np.clip(d["dN"], 1e-12, None)
        robust = d["a"] < d["a_uncertain"]
        ax.loglog(f[robust], dN[robust], color=col, lw=2.0, label=label)
        ax.loglog(f[~robust], dN[~robust], color=col, lw=1.6, ls=":")
        hh = bbh.hardening_rates(m1, m2, d["a"], rho, cs)
        ic = int(np.argmin(np.abs(np.abs(hh["dadt_gas"] / hh["dadt_gw"]) - 1.0)))
        ax.plot(f[ic], dN[ic], "o", color=col, ms=6, mec="k", mew=0.6, zorder=5)
        # structure-uncertain boundary, M_gas(<a)=M_BBH (curve is dotted to its left)
        # f_unc = float(physics.gw_frequency(m1, m2, d["a_uncertain"]))
        # if f_unc > 1e-4:
        #     ax.axvline(f_unc, color=col, ls=":", lw=1.2, ymin=0.33, ymax=0.82, alpha=0.9)

    ax.axhline(1.0, color="k", lw=0.8)
    ax.text(6e2, 1.45, r"$\delta N = 1$", fontsize=9, ha="right", va="bottom")
    ax.set_xlabel(r"GW frequency  $f$ [Hz]")
    ax.set_ylabel(r"dephasing to merger  $\delta N$ [cycles]")
    ax.set_xlim(1e-4, 1e3)
    ax.set_ylim(1e-6, 5e14)
    # legend (with a marker for the gas=GW crossover dots) moved off the lower-left inset
    handles, labels = ax.get_legend_handles_labels()
    xover = Line2D([], [], marker="o", ls="none", mfc="0.6", mec="k", mew=0.6, ms=6)
    ax.legend(handles + [xover], labels + [r"gas$=$GW crossover"],
              loc="lower center", bbox_to_anchor=(0.85, 0.5), fontsize=8.5,
              framealpha=0.95, borderpad=0.4)

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for j, (name, f0, f1) in enumerate(detband):
        y = 0.95 - 0.05 * j
        ax.plot([f0, f1], [y, y], transform=trans, color=DETC[name], lw=5.5,
                solid_capstyle="butt", clip_on=False)
        ax.text(np.sqrt(f0 * f1), y + 0.012, name, transform=trans, ha="center",
                va="bottom", fontsize=8, color=DETC[name], fontweight="bold")

   # ax.text(0.975, 0.525,
   #         r"$\bullet$ gas$=$GW crossover"  
   #         "\n" "(left: plunge, right: GW inspiral)",
            #"\n" r"vertical dotted: $M_{\rm gas}(<a){=}M_{\rm BBH}$"
            #r"  (curve dotted at lower $f$: structure-uncertain)",
   #         transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color="0.25")

    fig.tight_layout(pad=0.4)
    add_dN_chirp_inset(ax)        # schematic interpretation of delta N (added last, on top)
    return fig


# --------------------------------------------------------------------------- #
# Figure 6 : detectability (SNR) across bands
# --------------------------------------------------------------------------- #
def fig_snr(model, m1=10 * cst.MSUN, m2=10 * cst.MSUN, d_ref=100 * cst.MPC):
    rho_c, cs_c = model.rho_c, float(model.cs_of_r(model.r[1]))
    names = ["LISA", "DECIGO", "LVK"]

    fig, axs = plt.subplots(1, 2, figsize=(DCOL, 3.0),
                            gridspec_kw={"width_ratios": [1.45, 1]})

    D = np.logspace(np.log10(1 * cst.MPC), np.log10(2e4 * cst.MPC), 60)
    for name in names:
        s_gas = obs.snr(m1, m2, rho_c, cs_c, d_ref, name, gas=True)
        s_vac = obs.snr(m1, m2, rho_c, cs_c, d_ref, name, gas=False)
        axs[0].loglog(D / cst.MPC, s_gas * d_ref / D, color=DETC[name], lw=1.9, label=name)
        axs[0].loglog(D / cst.MPC, s_vac * d_ref / D, color=DETC[name], lw=1.1,
                      ls="--", alpha=0.7)
    axs[0].axhline(8, color="k", lw=0.8)
    axs[0].text(1.3, 9.5, "SNR = 8", fontsize=7.5)
    axs[0].set_xlabel("distance [Mpc]")
    axs[0].set_ylabel(r"matched-filter SNR")
    axs[0].set_ylim(1e-3, 1e6)
    axs[0].legend(loc="upper right", fontsize=8, title=r"$10{+}10\,M_\odot$",
                  title_fontsize=8)
    axs[0].text(0.035, 0.05, "solid: inside immortal\ndashed: vacuum",
                transform=axs[0].transAxes, fontsize=7, color="0.35", va="bottom")

    x = np.arange(len(names))
    t_gas = np.array([obs.time_in_band(m1, m2, rho_c, cs_c, det.BANDS[n], gas=True)
                      for n in names]) / cst.YR
    t_vac = np.array([obs.time_in_band(m1, m2, rho_c, cs_c, det.BANDS[n], gas=False)
                      for n in names]) / cst.YR
    w = 0.38
    axs[1].bar(x - w / 2, t_vac, w, color="0.72", label="vacuum")
    axs[1].bar(x + w / 2, t_gas, w, color=[DETC[n] for n in names], label="in immortal")
    axs[1].set_yscale("log")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names, fontsize=8)
    axs[1].set_ylabel("time in band [yr]")
    axs[1].legend(loc="upper right", fontsize=7.5)
    axs[1].grid(True, axis="y", which="major", alpha=0.25, lw=0.5)

    fig.tight_layout(pad=0.4)
    return fig
