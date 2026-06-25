"""Drag, accretion, gravitational-wave, and orbital-energy formulae.

Clean consolidation of ``CEE_functions.py``. All quantities are CGS. Functions
are vectorised over array inputs where it makes sense.

References
----------
Edgar 2004 (accretion review); Ostriker 1999, Kim & Kim 2007, Kim 2010
(dynamical friction in a gaseous medium); Peters 1964 (GW inspiral);
Cantiello et al. 2021 (AGN-star structure).
"""

import numpy as np

from . import constants as cst

# Avoid importing math.e; use numpy.
_E = np.e


# --------------------------------------------------------------------------- #
# Characteristic radii
# --------------------------------------------------------------------------- #
def schwarzschild_radius(m):
    """Schwarzschild radius 2GM/c^2 [cm]."""
    return 2.0 * cst.G * m / cst.C**2


def bondi_hoyle_radius(m, v, cs):
    """Bondi-Hoyle-Lyttleton radius 2GM/(v^2 + cs^2) [cm].

    Reduces to the HL radius 2GM/v^2 for supersonic motion (v >> cs).
    """
    return 2.0 * cst.G * m / (v**2 + cs**2)


def canto_rmin(m, v):
    """Inner cutoff r_min = sqrt(e) G m / (2 v^2) for the Coulomb logarithm.

    Canto et al. 2011 estimate of the perturber's effective size, used by the
    Kim (2010) dynamical-friction prescription.
    """
    return np.sqrt(_E) * cst.G * m / (2.0 * v**2)


# --------------------------------------------------------------------------- #
# Orbital kinematics / energetics
# --------------------------------------------------------------------------- #
def v_kepler(m1, m2, a):
    """Relative Keplerian velocity sqrt(G(m1+m2)/a) of a circular binary [cm/s]."""
    return np.sqrt(cst.G * (m1 + m2) / a)


def v_circular(m_enc, r):
    """Circular orbital speed sqrt(G M_enc(r)/r) about an enclosed mass [cm/s]."""
    return np.sqrt(cst.G * m_enc / r)


def orbital_energy(m1, m2, a):
    """Binding energy |E| = G m1 m2 / (2a) of a circular binary [erg] (positive)."""
    return cst.G * m1 * m2 / (2.0 * a)


def chirp_mass(m1, m2):
    """Chirp mass (m1 m2)^(3/5) / (m1+m2)^(1/5) [g]."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


# --------------------------------------------------------------------------- #
# Gravitational waves (circular, leading-order / Peters 1964)
# --------------------------------------------------------------------------- #
def gw_luminosity(m1, m2, a):
    """GW power radiated by a circular binary [erg/s] (Peters 1964)."""
    return (32.0 / 5.0) * (cst.G**4 / cst.C**5) * (m1 * m2) ** 2 * (m1 + m2) / a**5


def orbital_frequency(m1, m2, a):
    """Keplerian orbital frequency [Hz]."""
    return np.sqrt(cst.G * (m1 + m2) / a**3) / (2.0 * np.pi)


def gw_frequency(m1, m2, a):
    """Dominant GW frequency f_gw = 2 f_orb [Hz]."""
    return 2.0 * orbital_frequency(m1, m2, a)


def separation_from_gw_frequency(m1, m2, f_gw):
    """Invert f_gw = 2 f_orb for the separation a [cm]."""
    f_orb = f_gw / 2.0
    return (cst.G * (m1 + m2) / (2.0 * np.pi * f_orb) ** 2) ** (1.0 / 3.0)


def gw_strain(m1, m2, a, d_obs):
    """Sky/inclination-averaged GW strain amplitude h of a circular binary.

    Returns (f_gw, h). ``d_obs`` is the luminosity distance [cm].
    """
    mc = chirp_mass(m1, m2)
    f_gw = gw_frequency(m1, m2, a)
    h = (4.0 / d_obs) * (cst.G * mc / cst.C**2) * \
        (np.pi * f_gw * cst.G * mc / cst.C**3) ** (2.0 / 3.0)
    return f_gw, h


def t_merge_gw(m1, m2, a):
    """Time to coalescence of a circular binary from separation a [s] (Peters 1964).

    t = (5/256) c^5 a^4 / (G^3 m1 m2 (m1+m2)).
    """
    return (5.0 / 256.0) * cst.C**5 * a**4 / (cst.G**3 * m1 * m2 * (m1 + m2))


# --------------------------------------------------------------------------- #
# Accretion (Bondi-Hoyle-Lyttleton) and the associated momentum drag
# --------------------------------------------------------------------------- #
def bhl_accretion_rate(rho, m, cs, v):
    """Bondi-Hoyle-Lyttleton accretion rate [g/s] (Edgar 2004, eq. 32).

    Mdot = 4 pi G^2 rho m^2 / (cs^2 + v^2)^(3/2). Reduces to the HL rate
    4 pi G^2 rho m^2 / v^3 for v >> cs.
    """
    return 4.0 * np.pi * cst.G**2 * rho * m**2 / (cs**2 + v**2) ** 1.5


def accretion_drag_force(rho, m, cs, v):
    """Momentum drag from accretion, F = Mdot v [dyn] (Edgar 2004)."""
    return bhl_accretion_rate(rho, m, cs, v) * v


def accretion_drag_luminosity(rho, m, cs, v):
    """Power extracted from the orbit by accretion drag, F v = Mdot v^2 [erg/s]."""
    return bhl_accretion_rate(rho, m, cs, v) * v**2


def eddington_luminosity(m, kappa=0.34):
    """Eddington luminosity 4 pi G m c / kappa [erg/s]. Default kappa = 0.34."""
    return 4.0 * np.pi * cst.G * m * cst.C / kappa


def eddington_accretion_rate(m, eta=0.1, kappa=0.34):
    """Eddington-limited accretion rate L_Edd / (eta c^2) [g/s]."""
    return eddington_luminosity(m, kappa) / (eta * cst.C**2)


# --------------------------------------------------------------------------- #
# Geometric drag (legacy; kept for reference / comparison)
# --------------------------------------------------------------------------- #
def geometric_drag_luminosity(rho, R, v, C_d=1.0):
    """Aerodynamic drag power C_d rho pi R^2 v^3 [erg/s].

    This is the crude geometric drag used in the original trajectory notebook.
    The dynamical-friction prescription below is the physically appropriate
    treatment for a point mass moving through a gaseous medium; this is retained
    only for comparison.
    """
    return C_d * rho * np.pi * R**2 * v**3


# --------------------------------------------------------------------------- #
# Gaseous dynamical friction (Ostriker 1999 / Kim & Kim 2007 / Kim 2010)
# --------------------------------------------------------------------------- #
def coulomb_logarithm(mach, a, rmin):
    """Dimensionless drag coefficient I (the "Coulomb logarithm").

    Follows the piecewise fit collected by Schneider et al. 2024 / Bonner et
    al. 2024 (built on Kim 2010 + Kim & Kim 2007 + Ostriker 1999):

      * subsonic  (M < 1):    0.7706 ln[(1+M)/(1.0004-0.9185 M)] - 1.473 M
      * transonic (1 <= M < 4.4): ln[330 (a/rmin) (M-0.71)^5.72 / M^9.58]
      * supersonic (M >= 4.4): ln[(a/rmin)/(0.11 M + 1.65)]
    """
    mach = np.asarray(mach, dtype=float)
    a = np.broadcast_to(np.asarray(a, dtype=float), mach.shape)
    rmin = np.broadcast_to(np.asarray(rmin, dtype=float), mach.shape)

    out = np.zeros_like(mach)
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = mach < 1.0
        out[m1] = (0.7706 * np.log((1.0 + mach[m1]) / (1.0004 - 0.9185 * mach[m1]))
                   - 1.473 * mach[m1])
        m2 = (mach >= 1.0) & (mach < 4.4)
        out[m2] = np.log(330.0 * (a[m2] / rmin[m2])
                         * (mach[m2] - 0.71) ** 5.72 / mach[m2] ** 9.58)
        m3 = mach >= 4.4
        out[m3] = np.log((a[m3] / rmin[m3]) / (0.11 * mach[m3] + 1.65))
    # The transonic fit can go slightly negative just above M=1; clip at 0.
    return np.clip(out, 0.0, None)


def dynamical_friction_force(m_p, v_rel, a, cs, rho, rmin=None, C_d=1.0):
    """Gaseous dynamical-friction force on a perturber of mass m_p [dyn].

    Kim (2010) nonlinear prescription with the Kim & Kim (2007) Coulomb
    logarithm, as implemented for Bonner et al. 2024. ``a`` is the orbital
    radius of the perturber, ``rho``/``cs`` the local gas density/sound speed,
    ``v_rel`` the perturber speed, ``rmin`` the inner cutoff (default Canto+ 2011).
    """
    v_rel = np.asarray(v_rel, dtype=float)
    a = np.asarray(a, dtype=float)
    cs = np.asarray(cs, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if rmin is None:
        rmin = canto_rmin(m_p, v_rel)
    rmin = np.asarray(rmin, dtype=float)

    mach = v_rel / cs
    with np.errstate(divide="ignore", invalid="ignore"):
        B = cst.G * m_p / (a * cs**2)
        eta_b = B / (mach**2 - 1.0)
        f_rho = 1.0 + (0.46 * B**1.1) / (mach**2 - 1.0) ** 0.11
        I = coulomb_logarithm(mach, a, rmin)

        nonlinear = C_d * (0.7 / np.sqrt(np.abs(eta_b))) * 4.0 * np.pi * f_rho \
            * rho * (cst.G * m_p) ** 2 / v_rel**2
        linear = C_d * I * 4.0 * np.pi * rho * (cst.G * m_p) ** 2 / v_rel**2

    condition = (eta_b > 0.1) & (mach > 1.01)
    return np.where(condition, nonlinear, linear)


def dynamical_friction_luminosity(m_p, v_rel, a, cs, rho, rmin=None, C_d=1.0):
    """Power extracted from the orbit by gaseous dynamical friction, F v [erg/s]."""
    return dynamical_friction_force(m_p, v_rel, a, cs, rho, rmin, C_d) * v_rel


# --------------------------------------------------------------------------- #
# Gravitational binding energy of a stellar model
# --------------------------------------------------------------------------- #
def gamma1_mixture(beta):
    """Chandrasekhar (1939) first adiabatic exponent Gamma_1 for an ideal-gas + radiation
    mixture (gas gamma = 5/3). ``beta = P_gas/P_tot``; limits beta->1 give 5/3, beta->0 give 4/3.
    """
    beta = np.asarray(beta, dtype=float)
    return (32.0 - 24.0 * beta - 3.0 * beta**2) / (24.0 - 21.0 * beta)


def binding_energy_above(r, m_enc):
    """Gravitational binding energy U(>r) of the material outside radius r [erg].

    U(>r) = - integral_r^R  G M(r')/r'  dm,  dm = dM_enc.

    ``r`` and ``m_enc`` are 1-D arrays sorted by increasing radius (center to
    surface). Returns an array U(>r) <= 0 of the same length (the value at the
    outermost point is 0).

    Sanity check: for a uniform-density sphere this returns -3GM^2/(5R) at the
    center (see tests).
    """
    r = np.asarray(r, dtype=float)
    m_enc = np.asarray(m_enc, dtype=float)

    dm = np.diff(m_enc)                       # mass in each shell (length n-1)
    r_mid = 0.5 * (r[1:] + r[:-1])            # shell-centred radius
    m_mid = 0.5 * (m_enc[1:] + m_enc[:-1])    # shell-centred enclosed mass
    dU = -cst.G * m_mid / r_mid * dm          # energy of each shell

    U = np.zeros_like(r)
    # cumulative sum from the surface inward
    U[:-1] = np.cumsum(dU[::-1])[::-1]
    return U
