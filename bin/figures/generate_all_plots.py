#! /usr/bin/env python
"""
plot_survival_rates.py
"""

from plot_eta_eta_residual import generate_all_plots as plots0
from plot_process_priority import plot_star_process_priority as plots1
from plot_random_eta import generate_all_plots as plots2
from plot_survival_rates import plot_survival_rates as plots3
from plots_ulens_cands_samples import generate_all_plots as plots4
from generate_all_plots import generate_all_plots as plots5


if __name__ == '__main__':
    plots0()
    plots1()
    plots2()
    plots3()
    plots4()
    plots5()
