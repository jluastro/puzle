import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)


def generate_t_eff_array(t_eff_min=0.56, t_eff_max=550, t_eff_delta=0.333):
    t_eff = t_eff_min
    t_eff_arr = [t_eff]
    while True:
        t_eff *= 1 + t_eff_delta
        if t_eff > t_eff_max:
            break
        t_eff_arr.append(t_eff)
    return np.array(t_eff_arr)


def generate_t0_array(t_eff, t0_delta=0.333):
    t0_min = 58194.0 - t0_delta  # t0_delta less than first day of observation
    t0_max = 58848.0 + t0_delta  # t0_delta more than last day of observation

    t0 = t0_min
    t0_arr = [t0]
    while True:
        t0 += t_eff * t0_delta
        if t0 > t0_max:
            break
        t0_arr.append(t0)
    return np.array(t0_arr)


def plot_t0_t_eff_values():
    fig, ax = plt.subplots()
    t_eff_arr = generate_t_eff_array()
    for t_eff in t_eff_arr:
        t0_arr = generate_t0_array(t_eff)
        ax.scatter(t0_arr, np.ones(len(t0_arr)) * t_eff, s=5)
        ax.text(-20, t_eff, '%i' % len(t0_arr), fontsize=8, horizontalalignment='right')
    ax.set_xlim(-100, 1100)
    ax.set_yscale('log')
    ax.set_ylabel('t_eff')
    ax.set_xlabel('t0')


def return_q(t_arr, t0, t_eff):
    return 1 + ((t_arr - t0) / t_eff) ** 2.


def return_amplification_one(q):
    return q ** -0.5


def return_amplification_two(q):
    return (1 - ((q / 2) + 1) ** -2.) ** -0.5


def calculate_chi_squared(mag_obs_arr, mag_model_arr):
    assert len(mag_obs_arr) == len(mag_model_arr)
    num_obs = len(mag_obs_arr)
    numerator = (mag_obs_arr - mag_model_arr) ** 2.
    return np.sum(numerator / mag_model_arr) / num_obs


def fit_for_f0_f1(t_obs_arr, mag_obs_arr, t0, t_eff):
    return 0, 1


def fit_event(t_obs_arr, mag_obs_arr, mask_z=5, num_obs_min=50):
    t_eff_arr = generate_t_eff_array()
    for t_eff in t_eff_arr:
        t0_arr = generate_t0_array(t_eff)
        for t0 in t0_arr:
            f0, f1 = fit_for_f0_f1(t_obs_arr, mag_obs_arr, t0, t_eff)


            mask = t_obs_arr > t0 - mask_z * t_eff
            mask *= t_obs_arr < t0 + mask_z * t_eff
            if np.sum(mask) < num_obs_min:
                continue
            t_obs_masked = t_obs_arr[mask]
            mag_obs_masked = mag_obs_arr[mask]

            q = return_q(t_obs_masked, t0, t_eff)
            a_1 = return_amplification_one(q)
            chi_squared_1 = calculate_chi_squared(mag_obs_masked)
