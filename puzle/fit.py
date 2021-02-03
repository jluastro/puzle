import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import logging

from puzle.utils import return_data_dir

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


def ulens_func(a, f0, f1):
    return f1 * a + f0


def return_q(t_arr, t0, t_eff):
    return 1 + ((t_arr - t0) / t_eff) ** 2.


def return_amplification_one(q):
    return q ** -0.5


def return_amplification_two(q):
    return (1 - ((q / 2) + 1) ** -2.) ** -0.5


def return_chi_squared_mask(flux_obs, flux_model, percent_bad=10):
    assert len(flux_obs) == len(flux_model)
    chi_squared_arr = (flux_obs - flux_model) ** 2. / flux_model
    threshold = np.percentile(chi_squared_arr, 100 - percent_bad)
    mask = chi_squared_arr < threshold
    return mask


def calculate_chi_squared_model(flux_obs, flux_model):
    assert len(flux_obs) == len(flux_model)
    chi_squared_arr = (flux_obs - flux_model) ** 2. / flux_model
    return np.sum(chi_squared_arr) / len(chi_squared_arr)


def calculate_chi_squared_flat(flux_obs):
    chi_squared_arr = (flux_obs - np.mean(flux_obs)) ** 2. / np.mean(flux_obs)
    return np.sum(chi_squared_arr) / len(chi_squared_arr)


def _amplification(q, a_type):
    if a_type == 'one':
        a = return_amplification_one(q)
    elif a_type == 'two':
        a = return_amplification_two(q)
    else:
        raise Exception('a_type must be either "one" or "two"')
    return a


def return_amplification(t_arr, t0, t_eff, a_type):
    q = return_q(t_arr, t0, t_eff)
    a = _amplification(q, a_type)
    return a


def return_flux_model(t_arr, t0, t_eff, a_type, f0, f1):
    a = return_amplification(t_arr, t0, t_eff, a_type)
    return f1 * a + f0


def fit_event(t_obs_arr, flux_obs_arr, window_mask_z=5, num_obs_min=5):
    t0_total_arr = []
    t_eff_total_arr = []
    a_type_total_arr = []
    chi_squared_total_arr = []
    f0_total_arr = []
    f1_total_arr = []

    t_eff_arr = generate_t_eff_array()
    for t_eff in t_eff_arr:
        t0_arr = generate_t0_array(t_eff)
        for t0 in t0_arr:
            # apply a window mask
            window_mask = t_obs_arr > t0 - window_mask_z * t_eff
            window_mask *= t_obs_arr < t0 + window_mask_z * t_eff

            if np.sum(window_mask) < num_obs_min:
                continue

            for a_type in ['one', 'two']:
                # calculate the amplification array
                t_obs_masked = t_obs_arr[window_mask]
                flux_obs_masked = flux_obs_arr[window_mask]
                a = return_amplification(t_obs_masked, t0, t_eff, a_type)

                # take a first guess at f0 and f1
                bounds = ([-np.inf, 0],
                          [np.inf, np.inf])
                p0_f1 = np.max(flux_obs_masked) / np.max(a)
                p0_f0 = np.min(flux_obs_masked) - np.min(a * p0_f1)
                p0 = [p0_f0, p0_f1]
                (f0, f1), _ = curve_fit(ulens_func, a, flux_obs_masked,
                                        bounds=bounds, p0=p0)
                flux_model_masked = f1 * a + f0

                # remove the largest 10% chi squared data points, guess at f0 and f1
                chi_squared_mask = return_chi_squared_mask(flux_obs_masked,
                                                           flux_model_masked)
                t_obs_masked = t_obs_masked[chi_squared_mask]
                flux_obs_masked = flux_obs_masked[chi_squared_mask]
                a_masked = a[chi_squared_mask]
                (f0, f1), _ = curve_fit(ulens_func, a_masked, flux_obs_masked,
                                        bounds=bounds, p0=p0)
                flux_model_masked = f1 * a_masked + f0

                # calculate the increased chi_squared from model vs flat
                chi_squared_model = calculate_chi_squared_model(flux_obs_masked, flux_model_masked)
                chi_squared_flat = calculate_chi_squared_flat(flux_obs_masked)
                chi_squared_delta = ((chi_squared_model / chi_squared_flat) - 1) * len(flux_obs_masked)

                t0_total_arr.append(t0)
                t_eff_total_arr.append(t_eff)
                a_type_total_arr.append(a_type)
                chi_squared_total_arr.append(chi_squared_delta)
                f0_total_arr.append(f0)
                f1_total_arr.append(f1)

    return t0_total_arr, t_eff_total_arr, a_type_total_arr, chi_squared_total_arr, f0_total_arr, f1_total_arr


if __name__ == '__main__':
    fname = '%s/sample_ulens_lightcurve.npz' % return_data_dir()
    data = np.load(fname)
    t_obs_arr = data['hmjd']
    flux_obs_arr = data['flux']
    fluxerr_obs_arr = data['fluxerr']

    data = fit_event(t_obs_arr, flux_obs_arr)
    t0_arr, t_eff_arr, a_type_arr, chi_squared_arr, f0_arr, f1_arr = data

    idx = np.argmax(chi_squared_arr)
    a_type = a_type_arr[idx]
    t0 = t0_arr[idx]
    t_eff = t_eff_arr[idx]
    f0 = f0_arr[idx]
    f1 = f1_arr[idx]

    t_model_arr = np.arange(np.min(t_obs_arr), np.max(t_obs_arr), .1)
    flux_model_arr = return_flux_model(t_model_arr, t0, t_eff, a_type, f0, f1)

    fig, ax = plt.subplots()
    ax.scatter(t_obs_arr, flux_obs_arr, s=5, label='data', color='b')
    ax.plot(t_model_arr, flux_model_arr, label='model', color='r')
    ax.legend()