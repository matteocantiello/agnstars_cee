#!/usr/bin/env python3
"""Cross-check the local drag prescriptions against the stratified-wake drag of
Gagnier (2026, arXiv:2607.16422), reproducing the comparison quoted in the paper
(Sec. 3.3): the stratified azimuthal drag exceeds the local Kim & Kim (2007)
coefficient by a factor ~1-3 across the immortal's envelope, so the spiral-in
times in the paper are conservative upper estimates.

Requires the public DF_stratified package:

    git clone https://github.com/dgagnier/DF_stratified
    python scripts/gagnier_stratified_check.py --dfstrat ./DF_stratified

Notes on usage of DF_stratified (found necessary for a stellar model):
  * the radial domain must END at the stellar surface, where the outgoing-wave
    boundary condition applies (extending into the accretion stream turns the
    density cliff at the shock into a reflecting cavity with spurious sharp
    resonances);
  * the force at a single forcing frequency still sits on a comb of weak cavity
    resonances, so we quote the MEDIAN over a +/-6% Mach window (the sinking
    perturber sweeps frequency and never dwells on one eigenmode);
  * results are insensitive to the damping parameter eta over 1e-3..1e-2.
"""
import argparse
import sys
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfstrat", required=True, help="path to a DF_stratified clone")
    ap.add_argument("--lmax-cap", type=int, default=64)
    ap.add_argument("--eta", type=float, default=3e-3)
    args = ap.parse_args()

    sys.path.insert(0, args.dfstrat)
    try:
        from df_stratified.profiles import Profile
        from df_stratified.pde_solver import StratifiedWorkspace
    except ImportError as e:
        raise SystemExit(f"could not import df_stratified from {args.dfstrat!r}: {e}")

    from agn_cee import constants as cst, structure, physics, inspiral

    model = structure.build_model(structure.DEFAULT_PROFILE)
    Rstar = model.r_shock
    MBH = 10 * cst.MSUN

    # normalized stellar profile (units are irrelevant to the dimensionless force)
    r_grid = np.linspace(0.008, 1.0, 30000)                 # in R*, ends at the surface
    cs_ref = model.cs_gas_of_r(0.5 * Rstar)
    rho_n = model.rho_of_r(r_grid * Rstar)
    cs_n = model.cs_gas_of_r(r_grid * Rstar) / cs_ref

    def make_profile(a):
        return Profile(r=r_grid, rho=rho_n, cs=cs_n,
                       drho_dr=np.gradient(rho_n, r_grid, edge_order=2),
                       rho_at_a=float(np.interp(a, r_grid, rho_n)),
                       cs_at_a=float(np.interp(a, r_grid, cs_n)),
                       profile_name="immortal",
                       static_monopole_gauge="mass-conserving")

    def I_components(ws, mach):
        F = ws.force_one_mach(mach, args.eta)
        F0 = 4 * np.pi * ws.profile.rho_at_a / ws.profile.cs_at_a**2
        return np.sqrt(2) * F.real / (-F0), np.sqrt(2) * F.imag / (-F0)

    print("stratified vs local (KK07) azimuthal drag, 10 Msun BH on Keplerian orbits")
    print(f"{'a/R*':>6} {'M_K':>5} {'lmax':>5} {'Im(I)':>7} {'I_KK07':>7} {'ratio':>6}")
    rows = []
    for a_frac in (0.15, 0.30, 0.50, 0.70, 0.85):
        a_cm = a_frac * Rstar
        v = physics.v_circular(model.menc_of_r(a_cm), a_cm)
        cs_loc = model.cs_gas_of_r(a_cm)
        mach = v / cs_loc
        R_a = 2 * cst.G * MBH / (v**2 + cs_loc**2)
        lmax = int(min(args.lmax_cap, max(8, np.pi * a_cm / R_a)))
        b_min = np.pi * a_cm / lmax                          # scale resolved by the sum

        ws = StratifiedWorkspace(make_profile(a_frac), a=a_frac, lmax=lmax)
        t0 = time.time()
        vals = [I_components(ws, mm) for mm in mach * np.linspace(0.94, 1.06, 7)]
        im_med = float(np.median([x[1] for x in vals])) * mach**2
        I_kk07 = float(physics.coulomb_logarithm(mach, a_cm, b_min))
        rows.append((a_frac, im_med / I_kk07))
        print(f"{a_frac:6.2f} {mach:5.2f} {lmax:5d} {im_med:7.2f} {I_kk07:7.2f} "
              f"{im_med / I_kk07:6.2f}   [{time.time() - t0:.0f}s]")

    # impact on the single-CO spiral-in (outer-envelope leg dominates the total)
    a_arr = np.array([x[0] for x in rows])
    ratio = np.array([x[1] for x in rows])
    r_int = np.logspace(np.log10(0.098), 0.0, 400) * Rstar
    ch = inspiral.channel_luminosities(model, MBH, r_int)
    scale = np.interp(r_int / Rstar, a_arr, ratio)
    dJdr = np.gradient(MBH * ch["v_c"] * r_int, r_int)
    om = ch["v_c"] / r_int
    t0 = np.trapezoid(dJdr * om / (ch["L_df"] + ch["L_acc"]), r_int)
    t1 = np.trapezoid(dJdr * om / (ch["L_df"] * scale + ch["L_acc"]), r_int)
    print(f"\nspiral-in 0.1 R* -> R* leg: local {t0/cst.YR:.3f} yr -> "
          f"stratified-scaled {t1/cst.YR:.3f} yr (x{t1/t0:.2f})")


if __name__ == "__main__":
    main()
