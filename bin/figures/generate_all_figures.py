#! /usr/bin/env python
"""
generate_all_figures.py
"""

from plot_eta_eta_residual import generate_all_figures as figures0
from plot_process_priority import plot_star_process_priority as figures1
from plot_random_eta import generate_all_figures as figures2
from plot_survival_rates import plot_survival_rates as figures3
from plot_ulens_cands_samples import generate_all_figures as figures4
from plot_ulens_minmax_fits import generate_all_figures as figures5
from plot_ulens_opt_fits import generate_all_figures as figures6


if __name__ == '__main__':
    figures0()
    figures1()
    figures2()
    figures3()
    figures4()
    figures5()
    figures6()
