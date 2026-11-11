"""Sanity / physics tests for agn_cee. Run with: python -m pytest tests/ -q
or simply: python tests/test_physics.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agn_cee import constants as cst   # noqa: E402
from agn_cee import physics            # noqa: E402
from agn_cee import structure          # noqa: E402
from agn_cee import inspiral           # noqa: E402
from agn_cee import bbh                # noqa: E402
from agn_cee import observability as obs  # noqa: E402


def approx(a, b, rtol=1e-6):
    return np.allclose(a, b, rtol=rtol)


def test_schwarzschild_and_bondi():
    m = 10 * cst.MSUN
    rs = physics.schwarzschild_radius(m)
    assert approx(rs, 2 * cst.G * m / cst.C**2)
    # supersonic Bondi -> HL radius 2GM/v^2
    v, csnd = 1e9, 1e6
    assert approx(physics.bondi_hoyle_radius(m, v, csnd), 2 * cst.G * m / (v**2 + csnd**2))


def test_v_kepler_and_energy():
    m1 = m2 = 10 * cst.MSUN
    a = cst.RSUN
    v = physics.v_kepler(m1, m2, a)
    assert approx(v, np.sqrt(cst.G * (m1 + m2) / a))
    # virial: |E_orb| = (1/2) mu v^2 for equal masses? E = G m1 m2/2a
    assert approx(physics.orbital_energy(m1, m2, a), cst.G * m1 * m2 / (2 * a))


def test_gw_luminosity_and_merger_time():
    # Cross-check Peters merger time against energy/luminosity scaling.
    m1 = m2 = 10 * cst.MSUN
    a = 2 * cst.RSUN
    L = physics.gw_luminosity(m1, m2, a)
    # dE/dt = -L with E=-G m1 m2/2a gives da/dt=-64/5 G^3 m1 m2 M /(c^5 a^3).
    dadt = -2 * a**2 * L / (cst.G * m1 * m2)
    expected = -(64.0 / 5.0) * cst.G**3 * m1 * m2 * (m1 + m2) / (cst.C**5 * a**3)
    assert approx(dadt, expected, rtol=1e-10)
    # t_merge from integrating da/dt: t = a^4 / (4 * |coeff|) with coeff a^-3
    coeff = (64.0 / 5.0) * cst.G**3 * m1 * m2 * (m1 + m2) / cst.C**5
    t_int = a**4 / (4 * coeff)
    assert approx(physics.t_merge_gw(m1, m2, a), t_int, rtol=1e-10)


def test_gw_frequency_roundtrip():
    m1, m2 = 10 * cst.MSUN, 20 * cst.MSUN
    a = 3 * cst.RSUN
    f = physics.gw_frequency(m1, m2, a)
    assert approx(physics.separation_from_gw_frequency(m1, m2, f), a)


def test_binding_energy_uniform_sphere():
    # U_total of a uniform sphere = -3 G M^2 / (5 R).
    R = cst.RSUN
    M = cst.MSUN
    r = np.linspace(1e-3 * R, R, 20000)
    m_enc = M * (r / R) ** 3
    U = physics.binding_energy_above(r, m_enc)
    expected = -3.0 * cst.G * M**2 / (5.0 * R)
    assert approx(U[0], expected, rtol=2e-3), (U[0], expected)


def test_dynamical_friction_realistic():
    # Realistic in-star regime (Mach ~ a few): force is finite and positive.
    m = 10 * cst.MSUN
    rho, csnd, a = 1e-2, 1e8, 5 * cst.RSUN
    v = np.array([2e8, 3e8])  # Mach 2, 3
    F = physics.dynamical_friction_force(m, v, a=a, cs=csnd, rho=rho)
    assert np.all(np.isfinite(F)) and np.all(F > 0), F


def test_dynamical_friction_along_star():
    # Along the real structure the DF luminosity must be finite and non-negative.
    model = structure.build_model(structure.DEFAULT_PROFILE)
    m2 = 10 * cst.MSUN
    a = np.logspace(np.log10(2 * cst.RSUN), np.log10(model.r_shock), 200)
    rho = model.rho_of_r(a)
    csnd = model.cs_of_r(a)
    menc = model.menc_of_r(a)
    v = physics.v_circular(menc + m2, a)
    L = physics.dynamical_friction_luminosity(m2, v, a, csnd, rho)
    assert np.all(np.isfinite(L)) and np.all(L >= 0)
    assert np.any(L > 0)


def test_coulomb_logarithm_subsonic_matches_ostriker():
    # The KK07 subsonic fit must track the exact Ostriker (1999) result
    # I = 0.5 ln[(1+M)/(1-M)] - M in the mid-subsonic range (the fit is
    # intentionally finite at M -> 1, so it deviates there).
    mach = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    exact = 0.5 * np.log((1 + mach) / (1 - mach)) - mach
    fit = physics.coulomb_logarithm(mach, a=cst.RSUN, rmin=1e-3 * cst.RSUN)
    assert np.allclose(fit, exact, rtol=0.05), (fit, exact)
    # sharp regression pin at M=0.8: the published KK07 coefficient is 1.4703
    # (a 1.473 transcription would shift this value by ~0.7%)
    assert approx(float(physics.coulomb_logarithm(0.8, cst.RSUN, 1e-3 * cst.RSUN)),
                  0.29854, rtol=1e-3)


def test_eddington_luminosity_value():
    # L_Edd(1 Msun, kappa=0.34) = 4 pi G M c / kappa ~ 1.47e38 erg/s.
    L = physics.eddington_luminosity(cst.MSUN)
    assert 1.44e38 < L < 1.50e38, L


def test_bhl_rate_hl_limit_value():
    # HL limit (v >> cs): Mdot = 4 pi G^2 rho m^2 / v^3. For rho = 1e-9 g/cm^3,
    # m = 10 Msun, v = 1000 km/s this evaluates to 2.21e22 g/s.
    md = physics.bhl_accretion_rate(1e-9, 10 * cst.MSUN, 0.0, 1e8)
    assert approx(md, 2.215e22, rtol=1e-2), md


def test_gw_strain_value_bns():
    # Face-on strain of a 1.4+1.4 Msun binary at 100 Mpc, f_gw = 100 Hz:
    # h = (4/d)(G Mc/c^2)(pi f G Mc/c^3)^(2/3) ~ 3.56e-23 (optimal orientation).
    m = 1.4 * cst.MSUN
    a = physics.separation_from_gw_frequency(m, m, 100.0)
    f, h = physics.gw_strain(m, m, a, 100 * cst.MPC)
    assert approx(f, 100.0, rtol=1e-9)
    assert approx(h, 3.56e-23, rtol=2e-2), h


def test_structure_model_basic():
    model = structure.build_model(structure.DEFAULT_PROFILE)
    # mass and radius are physical
    assert 100 < model.m_star / cst.MSUN < 1000
    assert 5 < model.r_shock / cst.RSUN < 50
    # radius array strictly covers centre to Bondi and is increasing
    assert model.r[0] < model.r_shock < model.r[-1]
    assert np.all(np.diff(model.r) >= 0)
    # enclosed mass monotonic, reaches at least the stellar mass by the surface
    assert np.all(np.diff(model.m_enc) >= 0)
    i_shock = np.searchsorted(model.r, model.r_shock)
    assert model.m_enc[i_shock] / cst.MSUN > 100
    # binding energy negative and strongest at centre
    assert model.U_bind[0] < 0
    assert model.U_bind[0] <= model.U_bind[i_shock]


def test_bbh_vacuum_limit_recovers_peters():
    # With negligible gas density, gas hardening -> 0 and the merger time must
    # reduce to the Peters value.
    m1 = m2 = 10 * cst.MSUN
    a0 = cst.RSUN
    r = bbh.merger_time(m1, m2, a0, rho=1e-30, cs=1e8)
    assert approx(r["t_gas"], r["t_vacuum"], rtol=1e-3)
    assert approx(r["t_vacuum"], physics.t_merge_gw(m1, m2, a0), rtol=5e-2)


def test_bbh_gas_speeds_up_merger():
    # Real interior density must shorten the merger time vs vacuum.
    model = structure.build_model(structure.DEFAULT_PROFILE)
    m1 = m2 = 10 * cst.MSUN
    rho, cs = model.rho_c, float(model.cs_of_r(model.r[1]))
    r = bbh.merger_time(m1, m2, cst.RSUN, rho, cs)
    assert r["t_gas"] < r["t_vacuum"]
    assert r["ratio"] < 1.0


def test_bbh_dephasing_positive_and_monotone():
    # Dephasing (vac - gas cycles from f to merger) is non-negative and largest
    # at low frequency.
    model = structure.build_model(structure.DEFAULT_PROFILE)
    m1 = m2 = 10 * cst.MSUN
    d = bbh.dephasing(m1, m2, 3 * cst.RSUN, model.rho_c,
                      float(model.cs_of_r(model.r[1])))
    assert np.all(d["dN"] >= -1e-6)
    # dN increases toward low frequency (large separation)
    lowf = d["dN"][d["f_gw"] < 1e-3]
    highf = d["dN"][d["f_gw"] > 1e1]
    assert np.nanmax(lowf) > np.nanmax(highf)


def test_inspiral_time_decreases_with_mass():
    # A heavier compact object sinks faster (stronger drag).
    model = structure.build_model(structure.DEFAULT_PROFILE)
    t10 = inspiral.spiral_in_time(model, 10 * cst.MSUN, model.r_shock)["t_total"]
    t30 = inspiral.spiral_in_time(model, 30 * cst.MSUN, model.r_shock)["t_total"]
    assert t30 < t10


def test_snr_normalization_gw150914():
    # Sky-averaged single-detector O3 inspiral SNR of a GW150914-like binary at 410 Mpc
    # should match the observed ~13-24 (inspiral-only, so conservative for this high mass).
    s = obs.snr(35 * cst.MSUN, 30 * cst.MSUN, rho=1.0, cs=1e8,
                distance=410 * cst.MPC, detector="LVK", gas=False)
    assert 12 < s < 30, s


def test_snr_bns_range_calibration():
    # The single-detector O3 BNS (1.4+1.4) range (sky-averaged SNR=8 distance) should be
    # ~120 Mpc -- the gold-standard LVK sensitivity calibration.
    s100 = obs.snr(1.4 * cst.MSUN, 1.4 * cst.MSUN, 1.0, 1e8, 100 * cst.MPC, "LVK", gas=False)
    d_range = 100 * s100 / 8.0
    assert 90 < d_range < 160, d_range


def test_snr_inverse_distance():
    s1 = obs.snr(10 * cst.MSUN, 10 * cst.MSUN, 1.0, 1e8, 100 * cst.MPC, "LVK", gas=False)
    s2 = obs.snr(10 * cst.MSUN, 10 * cst.MSUN, 1.0, 1e8, 200 * cst.MPC, "LVK", gas=False)
    assert approx(s1 / s2, 2.0, rtol=1e-3)


def test_snr_callable_detector_guard():
    # Passing S_n as a callable must not silently drop the sky-average factor:
    # it requires an explicit sky_avg, and with the right factor it matches the
    # string-keyed path.
    from agn_cee import detectors
    args = (10 * cst.MSUN, 10 * cst.MSUN, 1.0, 1e8, 100 * cst.MPC)
    Sn_lvk = detectors.DETECTORS["LVK"]
    try:
        obs.snr(*args, Sn_lvk, gas=False)
        raise AssertionError("expected ValueError for callable detector without sky_avg")
    except ValueError:
        pass
    s_named = obs.snr(*args, "LVK", gas=False)
    s_callable = obs.snr(*args, Sn_lvk, gas=False, sky_avg=detectors.SKY_AVG["LVK"])
    assert approx(s_named, s_callable, rtol=1e-9)


def test_gas_suppresses_lisa_snr():
    # Inside the dense core the LISA-band SNR collapses relative to vacuum,
    # while the LVK SNR is essentially unchanged (merger is GW-dominated).
    model = structure.build_model(structure.DEFAULT_PROFILE)
    rho_c, cs_c = model.rho_c, float(model.cs_of_r(model.r[1]))
    m1 = m2 = 10 * cst.MSUN
    D = 100 * cst.MPC
    lisa_vac = obs.snr(m1, m2, rho_c, cs_c, D, "LISA", gas=False)
    lisa_gas = obs.snr(m1, m2, rho_c, cs_c, D, "LISA", gas=True)
    lvk_vac = obs.snr(m1, m2, rho_c, cs_c, D, "LVK", gas=False)
    lvk_gas = obs.snr(m1, m2, rho_c, cs_c, D, "LVK", gas=True)
    assert lisa_gas < 1e-2 * lisa_vac          # strong suppression in LISA
    assert approx(lvk_gas, lvk_vac, rtol=0.05)  # unaffected in LVK


def test_volumetric_rate_matches_paper():
    # The paper's fiducial chain gives R ~ 8 Gpc^-3 yr^-1 for n_BBH = 20/AGN/Myr.
    from agn_cee import rates
    assert approx(rates.volumetric_rate(20.0), 8.0, rtol=1e-6)
    assert approx(rates.migration_time() / rates.MYR, 0.5, rtol=1e-6)


def test_detection_rate_volume_cap():
    from agn_cee import rates
    big = 100 * rates.GPC
    uncapped = rates.detection_rate(8.0, big)
    capped = rates.detection_rate(8.0, big, V_max_Gpc3=150.0)
    assert capped < uncapped
    assert approx(capped, 8.0 * 150.0)


def test_review_gamma1_limits():
    from agn_cee import review
    assert approx(physics.gamma1_mixture(1.0), 5.0 / 3.0)
    assert approx(physics.gamma1_mixture(0.0), 4.0 / 3.0)
    # review must reuse the canonical implementation, not duplicate it
    assert review._gamma1_mixture is physics.gamma1_mixture


def test_review_soundspeed_ordering():
    # Radiation-dominated interior: cs_tot > cs_iso > cs_gas, and 0<beta<1.
    from agn_cee import review
    p = review.sound_speed_prescriptions()
    assert np.all((p["beta"] > 0) & (p["beta"] < 1))
    i = len(p["r"]) // 4   # a deep-interior point
    assert p["cs_tot"][i] > p["cs_iso"][i] > p["cs_gas"][i]


def test_review_net_binding_energy():
    # Net binding (grav - internal) is positive (bound) but well below |E_grav|.
    from agn_cee import review
    be = review.binding_energy_components()
    assert 0 < be["net_binding"] < be["omega"]
    assert 0.5 < be["ratio_int_over_grav"] < 1.0   # radiation-dominated -> E_int ~ |E_grav|


def test_review_component_focusing_supersonic():
    # Deep in the supersonic regime the focusing radius exceeds the orbital radius.
    from agn_cee import review
    cf = review.component_focusing(10 * cst.MSUN, 10 * cst.MSUN, 0.01 * cst.RSUN, cs=1.8e8)
    assert cf["mach1"] > 1 and cf["ratio1"] > 1


def test_review_hardening_ladder_brackets_baseline():
    # All ladder models harden faster than vacuum but slower than the independent
    # baseline (model 0), and none approach the vacuum (Peters) time.
    from agn_cee import review
    model = structure.build_model(structure.DEFAULT_PROFILE)
    cs = review.cs_at(review.sound_speed_prescriptions(), 0.5 * cst.RSUN, "cs_gas")
    m1 = m2 = 10 * cst.MSUN
    a0 = cst.RSUN
    t = {mdl: review.merger_time_model(m1, m2, a0, model.rho_c, cs, model=mdl) for mdl in range(5)}
    t_vac = physics.t_merge_gw(m1, m2, a0)
    assert all(t[mdl] < 1e-2 * t_vac for mdl in range(5))     # all >> faster than vacuum
    assert all(t[mdl] >= t[0] * 0.99 for mdl in range(5))     # model 0 is the fastest (upper drag)
    assert t[2] / t[0] < 5                                     # companion-only correction ~ factor few


def test_review_companion_wake_fraction_bounds():
    from agn_cee import review
    import numpy as _np
    assert approx(float(review.companion_wake_fraction(10.0)), 0.45, rtol=1e-6)  # supersonic
    assert float(review.companion_wake_fraction(0.2)) > 0.45                     # subsonic stronger
    assert _np.all(review.companion_wake_fraction(_np.array([0.1, 1.0, 5.0])) <= 0.95)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)


def test_fallback_mesa_reader_parses_header():
    # Without mesa_reader installed (it is not in requirements.txt), the built-in reader
    # must still expose the header values the notebook uses (star_mdot, photosphere_L).
    from agn_cee import structure
    p = structure._MesaProfileFallback(structure.DEFAULT_PROFILE)
    assert abs(p.star_mass - 773.3) < 0.1           # header: total mass [Msun]
    assert 0.0 < p.star_mdot < 1.0                  # header: accretion rate [Msun/yr]
    assert p.photosphere_L > 1e7                     # header: luminosity [Lsun]
    assert abs(p.mass[0] - p.star_mass) < 1e-6       # zone table agrees with header
