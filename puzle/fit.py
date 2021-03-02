from scipy.optimize import curve_fit
import numpy as np
import logging
from zort.photometry import magnitudes_to_fluxes

from puzle.utils import return_data_dir, load_stacked_arrays

logger = logging.getLogger(__name__)


def return_q(t_arr, t0, t_eff):
    return 1 + (((t_arr - t0) / t_eff)*((t_arr - t0) / t_eff))


def return_amplification_one(q):
    return 1 / np.sqrt(q)


def return_amplification_two(q):
    return 1 / np.sqrt((1 - (1 / (((q / 2) + 1)*((q / 2) + 1)))))


def return_chi_squared_mask(flux_obs, fluxerr_obs, flux_model, percent_bad=10):
    assert len(flux_obs) == len(flux_model)
    assert len(flux_obs) == len(fluxerr_obs)
    chi_squared_arr = ((flux_obs - flux_model) ** 2.) / (fluxerr_obs ** 2.)
    threshold = np.percentile(chi_squared_arr, 100 - percent_bad)
    mask = chi_squared_arr < threshold
    return mask


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


def ulens_func_a_type_one(t_arr, t0, t_eff, f0, f1):
    q = return_q(t_arr, t0, t_eff)
    a = return_amplification_one(q)
    return f1 * a + f0


def ulens_func_a_type_two(t_arr, t0, t_eff, f0, f1):
    q = return_q(t_arr, t0, t_eff)
    a = return_amplification_two(q)
    return f1 * a + f0


def fit_event(t_obs_arr, mag_arr, magerr_arr):
    flux_obs_arr, fluxerr_obs_arr = magnitudes_to_fluxes(mag_arr, magerr_arr)
    bounds = ([np.min(t_obs_arr) - 50, 0.01, -np.inf, 0],
              [np.max(t_obs_arr) + 50, 1000, np.inf, np.inf])

    t0_arr = []
    t_eff_arr = []
    f0_arr = []
    f1_arr = []
    chi_squared_delta_arr = []
    a_type_arr = []
    for a_type in ['one', 'two']:
        if a_type == 'one':
            ulens_func = ulens_func_a_type_one
            amplification_func = return_amplification_one
        elif a_type == 'two':
            ulens_func = ulens_func_a_type_two
            amplification_func = return_amplification_two

        try:
            (t0, t_eff, f0, f1), _ = curve_fit(ulens_func,
                                               t_obs_arr, flux_obs_arr,
                                               bounds=bounds)
        except RuntimeError:
            continue
        q = return_q(t_obs_arr, t0, t_eff)
        a = amplification_func(q)
        flux_model_arr = f1 * a + f0

        chi_squared_model = np.sum((flux_obs_arr - flux_model_arr) ** 2. / fluxerr_obs_arr ** 2.)
        chi_squared_flat = np.sum((flux_obs_arr - np.mean(flux_obs_arr)) ** 2. / fluxerr_obs_arr ** 2.)
        # chi_squared_delta = ((chi_squared_model / chi_squared_flat) - 1) * len(flux_model_arr)
        chi_squared_delta = chi_squared_flat - chi_squared_model

        t0_arr.append(t0)
        t_eff_arr.append(t_eff)
        f0_arr.append(f0)
        f1_arr.append(f1)
        chi_squared_delta_arr.append(chi_squared_delta)
        a_type_arr.append(a_type)

    if len(t0_arr) == 0:
        return None

    idx = np.argmin(chi_squared_delta_arr)

    t0 = t0_arr[idx]
    t_eff = t_eff_arr[idx]
    f0 = f0_arr[idx]
    f1 = f1_arr[idx]
    chi_squared_delta = chi_squared_delta_arr[idx]
    a_type = a_type_arr[idx]

    return t0, t_eff, f0, f1, chi_squared_delta, a_type


def main():
    fname = '%s/ulens_sample.npz' % return_data_dir()
    lightcurves_arr = load_stacked_arrays(fname)
    chi_squared_delta_arr = []
    for i, lightcurve in enumerate(lightcurves_arr):
        if i % 10 == 0:
            print('Fitting lightcurve %i / %i' % (i, len(lightcurves_arr)))
        t_obs_arr = lightcurve[:, 0]
        mag_arr = lightcurve[:, 1]
        magerr_arr = lightcurve[:, 2]

        data = fit_event(t_obs_arr, mag_arr, magerr_arr)
        if data is None:
            chi_squared_delta_arr.append(0)
        else:
            t0, t_eff, f0, f1, chi_squared_delta, a_type = data
            chi_squared_delta_arr.append(chi_squared_delta)
    return chi_squared_delta_arr


if __name__ == '__main__':
    main()
