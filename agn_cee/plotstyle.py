"""Publication-grade matplotlib styling (AAS journals; no MESA/seaborn deps).

Absolute point sizes are used so that every figure has the same physical text
size when included at its natural width. Figure widths follow the AAS template:
``COL`` (3.4 in) for single-column figures, ``DCOL`` (7.1 in) for double-column.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# AAS figure widths [inches]
COL = 3.4     # \columnwidth
DCOL = 7.1    # \textwidth

# Okabe-Ito colourblind-safe palette
BLACK = (0.00, 0.00, 0.00)
ORANGE = (0.90, 0.60, 0.00)
SKY_BLUE = (0.35, 0.70, 0.90)
BLUE_GREEN = (0.00, 0.60, 0.50)
YELLOW = (0.95, 0.90, 0.25)
BLUE = (0.00, 0.45, 0.70)
VERMILLION = (0.80, 0.40, 0.00)
RED_PURPLE = (0.80, 0.60, 0.70)
COLORS = [ORANGE, BLUE_GREEN, SKY_BLUE, RED_PURPLE, VERMILLION, YELLOW, BLUE, BLACK]

# Consistent semantic colours used across figures.
DENSITY_COLORS = [BLUE, BLUE_GREEN, ORANGE]          # core, mid-envelope, outer envelope
DETECTOR_COLORS = {"LISA": SKY_BLUE, "DECIGO": RED_PURPLE, "LVK": VERMILLION}


def apply():
    """Apply the publication style and return pyplot."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "patch.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0, "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2, "ytick.minor.size": 2.2,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.labelspacing": 0.3,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "axes.formatter.use_mathtext": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return plt
