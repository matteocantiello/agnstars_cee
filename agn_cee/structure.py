"""Immortal-star structure model: MESA interior + analytic accretion stream.

The interior (hydrostatic, from the stellar centre to the accretion shock at the
stellar surface) is read from a MESA profile. We take the quantities MESA reports
directly -- the **mass coordinate** M(<r), the total pressure, density, temperature
and mean molecular weight -- rather than reconstructing them, because the
4 pi r^2 rho dr mass reconstruction is ~10-20% high in the inner region.

Outside the shock, the accretion stream is described by the analytic Bondi-like
scalings of Cantiello et al. (2021): rho ~ r^-3/2, T ~ r^-3/8, v ~ r^-1/2,
normalised so that at the Bondi radius the density, sound speed and inflow velocity
match the local AGN-disk conditions, and the temperature matches the stellar
surface at the shock.

Sound speed. We expose three prescriptions, computed from the MESA pressure:
    cs_gas = sqrt(5/3 P_gas/rho)      gas-only (ideal-gas) -- the FIDUCIAL,
    cs_tot = sqrt(Gamma_1 P_tot/rho)  full adiabatic (gas + radiation),
    cs_iso = sqrt(P_tot/rho)          isothermal-total (conservative middle),
with P_gas = P_tot - P_rad, P_rad = a T^4/3, and Gamma_1 the Chandrasekhar mixture
value. ``model.cs`` / ``model.cs_of_r`` return the fiducial gas-only speed, so every
calculation that reads them (Case A, Case B, dephasing, strain) uses one consistent
c_s; ``cs_tot_of_r`` / ``cs_iso_of_r`` give the alternatives for the reported brackets.

The model also exposes the interior gravitational binding energy |Omega|, internal
energy E_int and net binding |Omega|-E_int (MESA mass column, hydrostatic star only,
excluding the spherically-integrated accretion stream).
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from . import constants as cst
from . import physics


GAMMA_GAS = 5.0 / 3.0     # ideal-gas adiabatic index
CS_AGN_DEFAULT = 1.0e6    # AGN-disk sound speed at the star's location [cm/s]
RHO_AGN_DEFAULT = 3.0e-17  # AGN-disk density at the star's location [g/cm^3]
A_RAD = 4.0 * cst.SIGMA_SB / cst.C   # radiation constant a = 4 sigma/c [erg cm^-3 K^-4]


@dataclass
class StarStreamModel:
    """Container for the immortal-star + accretion-stream structure (CGS).

    Arrays run from the stellar centre outward to the Bondi radius.
    """

    r: np.ndarray            # radius [cm]
    rho: np.ndarray          # density [g/cm^3]
    cs: np.ndarray           # fiducial (gas-only) sound speed [cm/s]
    cs_gas: np.ndarray       # gas-only sound speed [cm/s] (= cs)
    cs_tot: np.ndarray       # full adiabatic (gas+radiation) sound speed [cm/s]
    cs_iso: np.ndarray       # isothermal-total sound speed [cm/s]
    T: np.ndarray            # temperature [K]
    v_stream: np.ndarray     # accretion-stream inflow speed [cm/s] (0 inside star)
    m_enc: np.ndarray        # enclosed mass [g] (MESA column inside the star)
    U_bind: np.ndarray       # binding energy of material outside r [erg] (<= 0)

    # scalars
    m_star: float            # stellar (interior) mass [g]
    r_shock: float           # shock / stellar surface radius [cm]
    r_bondi: float           # Bondi radius [cm]
    rho_c: float             # central density [g/cm^3]
    cs_agn: float            # AGN sound speed [cm/s]
    rho_agn: float           # AGN density [g/cm^3]
    omega_bind: float        # interior gravitational binding energy |Omega| [erg] (>0)
    E_int: float             # interior internal energy [erg] (>0)
    net_binding: float       # |Omega| - E_int [erg]

    def __post_init__(self):
        def clamp(y):
            # clamp to the boundary value outside the grid (NO linear extrapolation):
            # below the inner grid -> central value, beyond the Bondi radius -> outer value.
            return interp1d(self.r, y, kind="linear", assume_sorted=True,
                            bounds_error=False, fill_value=(y[0], y[-1]))
        self.rho_of_r = clamp(self.rho)
        self.cs_of_r = clamp(self.cs)
        self.cs_gas_of_r = clamp(self.cs_gas)
        self.cs_tot_of_r = clamp(self.cs_tot)
        self.cs_iso_of_r = clamp(self.cs_iso)
        self.menc_of_r = clamp(self.m_enc)
        self.vstream_of_r = clamp(self.v_stream)
        self.Ubind_of_r = clamp(self.U_bind)


def load_mesa_profile(path):
    """Load a MESA profile via mesa_reader (falls back to a direct parser)."""
    try:
        import mesa_reader as mr
        return mr.MesaData(path)
    except Exception:
        return _MesaProfileFallback(path)


class _MesaProfileFallback:
    """Minimal MESA-profile reader (header line 6 = names, data from line 7)."""

    def __init__(self, path):
        names = None
        rows = []
        with open(path) as fh:
            for i, line in enumerate(fh, start=1):
                if i == 6:
                    names = line.split()
                elif i >= 7 and line.strip():
                    rows.append([float(x) for x in line.split()])
        data = np.array(rows)
        self._data = {n: data[:, j] for j, n in enumerate(names)}

    def __getattr__(self, name):
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def build_model(profile_path, cs_agn=CS_AGN_DEFAULT, rho_agn=RHO_AGN_DEFAULT,
                n_stream=1000):
    """Build the immortal-star + accretion-stream model from a MESA profile.

    Parameters
    ----------
    profile_path : str
        Path to the MESA ``profile*.data`` file.
    cs_agn, rho_agn : float
        AGN-disk sound speed [cm/s] and density [g/cm^3] at the star's location.
    n_stream : int
        Number of points sampling the accretion stream.
    """
    p = load_mesa_profile(profile_path)

    # --- interior (MESA), ordered centre -> surface; quantities taken DIRECTLY ---
    r_int = (10.0 ** p.logR)[::-1] * cst.RSUN     # radius [cm]
    rho_int = (10.0 ** p.logRho)[::-1]            # density [g/cm^3]
    T_int = (10.0 ** p.logT)[::-1]                # temperature [K]
    P_int = (10.0 ** p.logP)[::-1]                # total pressure [dyn/cm^2]
    mu_int = np.asarray(p.mu, dtype=float)[::-1]  # mean molecular weight
    m_enc_int = np.asarray(p.mass, dtype=float)[::-1] * cst.MSUN   # MESA mass coordinate M(<r)
    v_int = np.zeros_like(r_int)                  # interior is hydrostatic

    m_star = p.mass[0] * cst.MSUN                 # total interior mass [g]
    r_shock = (10.0 ** p.logR[0]) * cst.RSUN      # stellar surface = shock radius
    T_shock = (10.0 ** p.logT[0])                 # surface temperature
    r_bondi = 2.0 * cst.G * m_star / cs_agn**2

    # interior sound speeds from the MESA pressure (gas / adiabatic / isothermal-total)
    P_rad_int = A_RAD * T_int**4 / 3.0
    P_gas_int = np.clip(P_int - P_rad_int, 1e-30, None)
    g1_int = physics.gamma1_mixture(P_gas_int / P_int)
    csg_int = np.sqrt(5.0 / 3.0 * P_gas_int / rho_int)
    cst_int = np.sqrt(g1_int * P_int / rho_int)
    csi_int = np.sqrt(P_int / rho_int)

    # --- accretion stream (analytic), ordered shock -> Bondi ---
    r_str = np.linspace(r_shock, r_bondi, n_stream)
    x = r_str / cst.RSUN
    rho_str = x ** (-1.5)
    T_str = x ** (-0.375)
    v_str = x ** (-0.5)
    # normalise: rho(r_bondi)=rho_agn, T(r_shock)=T_shock, v(r_bondi)=cs_agn
    rho_str *= rho_agn / rho_str[-1]
    T_str *= T_shock / T_str[0]
    v_str *= cs_agn / v_str[-1]
    # stream gas pressure from the ideal-gas law with the surface mean molecular weight,
    # which makes cs_gas continuous across the shock (P_gas,interior = rho kB T / mu mp).
    mu_surf = float(mu_int[-1])
    P_gas_str = rho_str * cst.KB * T_str / (mu_surf * cst.MP)
    P_rad_str = A_RAD * T_str**4 / 3.0
    P_tot_str = P_gas_str + P_rad_str
    g1_str = physics.gamma1_mixture(P_gas_str / P_tot_str)
    csg_str = np.sqrt(5.0 / 3.0 * P_gas_str / rho_str)
    cst_str = np.sqrt(g1_str * P_tot_str / rho_str)
    csi_str = np.sqrt(P_tot_str / rho_str)
    # stream enclosed mass: M_star plus the cumulative (analytic) stream shell mass
    dr_str = np.diff(r_str)
    dr_str = np.insert(dr_str, 0, dr_str[0])
    m_enc_str = m_star + np.cumsum(4.0 * np.pi * rho_str * r_str**2 * dr_str)

    # --- concatenate (centre -> Bondi) ---
    r = np.concatenate((r_int, r_str))
    rho = np.concatenate((rho_int, rho_str))
    T = np.concatenate((T_int, T_str))
    v_stream = np.concatenate((v_int, v_str))
    m_enc = np.concatenate((m_enc_int, m_enc_str))
    cs_gas = np.concatenate((csg_int, csg_str))
    cs_tot = np.concatenate((cst_int, cst_str))
    cs_iso = np.concatenate((csi_int, csi_str))
    cs = cs_gas                                   # fiducial = gas-only

    # --- binding energy of the hydrostatic interior (MESA mass column, no stream) ---
    dM = np.diff(m_enc_int, prepend=0.0)
    omega_bind = float(np.sum(cst.G * m_enc_int / np.maximum(r_int, 1.0) * dM))   # |Omega| > 0
    u_int = 1.5 * P_gas_int + 3.0 * P_rad_int                                     # u = 3/2 P_gas + 3 P_rad
    E_int = float(np.sum(u_int / rho_int * dM))
    net_binding = omega_bind - E_int
    # binding-energy-above profile (illustrative curve), computed with the corrected M(r)
    U_bind = physics.binding_energy_above(r, m_enc)

    return StarStreamModel(
        r=r, rho=rho, cs=cs, cs_gas=cs_gas, cs_tot=cs_tot, cs_iso=cs_iso,
        T=T, v_stream=v_stream, m_enc=m_enc, U_bind=U_bind,
        m_star=m_star, r_shock=r_shock, r_bondi=r_bondi, rho_c=rho_int[0],
        cs_agn=cs_agn, rho_agn=rho_agn,
        omega_bind=omega_bind, E_int=E_int, net_binding=net_binding,
    )


# Default profile shipped in the repo.
DEFAULT_PROFILE = "data/14547_cs10_RHO_3d-17_profile20.data"
