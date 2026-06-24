#!/usr/bin/env python3
"""Regenerate every paper figure into ./figures/ (non-interactive).

Run from the repository root:

    python make_figures.py

The output filenames match the ``\\includegraphics`` calls in the manuscript;
the third column is the figure's number in the compiled paper.
"""
import os

from agn_cee import structure, figures, review

OUT = "figures"

# (output filename, builder, label-in-paper)
FIGS = [
    ("fig1_structure.pdf",             figures.fig_structure,            "Fig. 1"),
    ("fig3_caseA_inspiral.pdf",        figures.fig_inspiral,             "Fig. 2"),
    ("fig9_spiral_trajectories.pdf",   review.fig_true_trajectories,     "Fig. 3"),
    ("fig2_caseA_powers.pdf",          figures.fig_power,                "Fig. 4"),
    ("fig4_caseB_powers.pdf",          review.fig_power_caseB,           "Fig. 5"),
    ("fig8_energy_deposition.pdf",     review.fig_energy_deposition,     "Fig. 6"),
    ("fig6_dephasing.pdf",             figures.fig_dephasing,            "Fig. 7"),
    ("fig7_characteristic_strain.pdf", review.fig_characteristic_strain, "Fig. 8"),
    ("fig5_hardening_ladder.pdf",      review.fig_hardening_ladder,      "Fig. 9 (App. B)"),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    model = structure.build_model(structure.DEFAULT_PROFILE)
    print("Built immortal-star model: M_* = %.0f Msun, R_shock = %.1f Rsun, rho_c = %.1f g/cm^3"
          % (model.m_star / 1.989e33, model.r_shock / 6.955e10, model.rho_c))
    for fname, builder, label in FIGS:
        fig = builder(model)
        path = os.path.join(OUT, fname)
        fig.savefig(path)
        figures.plt.close(fig)
        print("  %-16s -> %s" % (label, path))


if __name__ == "__main__":
    main()
