"""Immortal-star structure model: MESA interior + analytic accretion stream.

The interior (hydrostatic, from the stellar centre to the accretion shock at the
stellar surface) is read from a MESA profile. Outside the shock, the accretion
stream is described by the analytic Bondi-like scalings of Cantiello et al.
(2021): rho ~ r^-3/2, T ~ r^-3/8, v ~ r^-1/2, normalised so that at the Bondi
radius the density, sound speed and inflow velocity match the local AGN-disk
conditions, and the temperature matches the stellar surface at the shock.

The resulting model exposes density, sound speed, enclosed mass and stream
velocity as functions of radius (interpolators), plus the binding-energy
profile of the envelope.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d

from . import constants as cst
from . import physics


GAMMA_GAS = 5.0 / 3.0     # adiabatic index used for the sound speed
CS_AGN_DEFAULT = 1.0e6    # AGN-disk sound speed at the star's location [cm/s]
RHO_AGN_DEFAULT = 3.0e-17  # AGN-disk density at the star's location [g/cm^3]


@dataclass
class StarStreamModel:
    """Container for the immortal-star + accretion-stream structure (CGS).

    Arrays run from the stellar centre outward to the Bondi radius.
    """

    r: np.ndarray            # radius [cm]
    rho: np.ndarray          # density [g/cm^3]
    cs: np.ndarray           # sound speed [cm/s]
    T: np.ndarray            # temperature [K]
    v_stream: np.ndarray     # accretion-stream inflow speed [cm/s] (0 inside star)
    m_enc: np.ndarray        # enclosed mass [g]
    U_bind: np.ndarray       # binding energy of material outside r [erg] (<= 0)

    # scalars
    m_star: float            # stellar (interior) mass [g]
    r_shock: float           # shock / stellar surface radius [cm]
    r_bondi: float           # Bondi radius [cm]
    rho_c: float             # central density [g/cm^3]
    cs_agn: float            # AGN sound speed [cm/s]
    rho_agn: float           # AGN density [g/cm^3]

    def __post_init__(self):
        kw = dict(kind="linear", assume_sorted=True,
                  bounds_error=False, fill_value="extrapolate")
        self.rho_of_r = interp1d(self.r, self.rho, **kw)
        self.cs_of_r = interp1d(self.r, self.cs, **kw)
        self.menc_of_r = interp1d(self.r, self.m_enc, **kw)
        self.vstream_of_r = interp1d(self.r, self.v_stream, **kw)
        self.Ubind_of_r = interp1d(self.r, self.U_bind, **kw)


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
                gamma=GAMMA_GAS, n_stream=1000):
    """Build the immortal-star + accretion-stream model from a MESA profile.

    Parameters
    ----------
    profile_path : str
        Path to the MESA ``profile*.data`` file.
    cs_agn, rho_agn : float
        AGN-disk sound speed [cm/s] and density [g/cm^3] at the star's location.
    gamma : float
        Adiabatic index for cs = sqrt(gamma kB T / mp).
    n_stream : int
        Number of points sampling the accretion stream.
    """
    p = load_mesa_profile(profile_path)

    # --- interior (MESA), ordered centre -> surface ---
    r_int = (10.0 ** p.logR)[::-1] * cst.RSUN     # [cm]
    rho_int = (10.0 ** p.logRho)[::-1]            # [g/cm^3]
    T_int = (10.0 ** p.logT)[::-1]                # [K]
    v_int = np.zeros_like(r_int)                  # interior is hydrostatic

    m_star = p.mass[0] * cst.MSUN                 # total interior mass [g]
    r_shock = (10.0 ** p.logR[0]) * cst.RSUN      # stellar surface = shock radius
    T_shock = (10.0 ** p.logT[0])                 # surface temperature
    r_bondi = 2.0 * cst.G * m_star / cs_agn**2

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

    # --- concatenate ---
    r = np.concatenate((r_int, r_str))
    rho = np.concatenate((rho_int, rho_str))
    T = np.concatenate((T_int, T_str))
    v_stream = np.concatenate((v_int, v_str))
    cs = np.sqrt(gamma * T * cst.KB / cst.MP)

    # --- enclosed mass (integrate 4 pi rho r^2 dr outward) ---
    dr = np.diff(r)
    dr = np.insert(dr, 0, dr[0])   # first shell uses the first spacing (avoids M=0)
    dm = 4.0 * np.pi * rho * r**2 * dr
    m_enc = np.cumsum(dm)

    # --- binding energy of the envelope above each radius ---
    U_bind = physics.binding_energy_above(r, m_enc)

    return StarStreamModel(
        r=r, rho=rho, cs=cs, T=T, v_stream=v_stream, m_enc=m_enc, U_bind=U_bind,
        m_star=m_star, r_shock=r_shock, r_bondi=r_bondi, rho_c=rho_int[0],
        cs_agn=cs_agn, rho_agn=rho_agn,
    )


# Default profile shipped in the repo.
DEFAULT_PROFILE = "data/14547_cs10_RHO_3d-17_profile20.data"
