#! /usr/bin/env python
"""
OLD_calculate_obj_snr_cuts.py
"""
import os
import glob
import numpy as np
from zort.lightcurveFile import LightcurveFile
from zort.radec import load_ZTF_fields
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from puzle.utils import return_figures_dir, return_DR5_dir


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def return_CDF(arr):
    x = np.sort(arr)
    y = np.arange(len(arr)) / (len(arr) - 1)
    return x, y


def return_high_field_ids():
    ZTF_fields = load_ZTF_fields()
    cond = (ZTF_fields['galLong'] >= 4.5) * (ZTF_fields['galLong'] <= 20)
    cond *= (ZTF_fields['galLat'] >= -10) * (ZTF_fields['galLat'] <= -2)
    field_ids = ZTF_fields[cond]['id']
    idx = np.argsort(field_ids)

    field_ids = ZTF_fields[cond]['id'][idx]
    glons = ZTF_fields[cond]['galLong'][idx]
    glats = ZTF_fields[cond]['galLat'][idx]
    return field_ids, glons, glats


def return_low_field_ids():
    ZTF_fields = load_ZTF_fields()
    cond = (ZTF_fields['galLong'] >= 40) * (ZTF_fields['galLong'] <= 60)
    cond *= (ZTF_fields['galLat'] >= 2) * (ZTF_fields['galLat'] <= 7)
    field_ids = ZTF_fields[cond]['id']
    idx = np.argsort(field_ids)

    field_ids = ZTF_fields[cond]['id'][idx]
    glons = ZTF_fields[cond]['galLong'][idx]
    glats = ZTF_fields[cond]['galLat'][idx]
    return field_ids, glons, glats


def generate_obj_snr_cuts(field_ids, glons, glats, N_samples=10000):
    DR5_dir = return_DR5_dir()
    for field_id, glon, glat in zip(field_ids, glons, glats):
        filenames = glob.glob('%s/field%06d_*txt' % (DR5_dir, field_id))
        if len(filenames) != 1:
            continue
        filename = filenames[0]
        print('Processing %s' % filename)

        object_filename = filename.replace('.txt', '.objects')

        n_objects = file_len(object_filename)
        process_size = (n_objects // N_samples) + 1
        lightcurveFile = LightcurveFile(filename,
                                        proc_rank=0, proc_size=process_size)
        snr_arr = []
        for obj in lightcurveFile:
            if obj.lightcurve.nepochs < 20:
                continue
            magerr = obj.lightcurve.magerr
            snr = np.median(1.0875/magerr)
            if snr < 0:
                continue
            snr_arr.append(snr)

        print('-- %i samples collected' % len(snr_arr))
        if len(snr_arr) == 0:
            continue

        bins = np.arange(5, 50)
        fig, ax = plt.subplots(2, 1)
        fig.suptitle('Field %i | %i Samples' % (field_id, len(snr_arr)), fontsize=12)

        ax[0].set_title('(l, b) = (%.1f, %.1f)' % (glon, glat), fontsize=12)
        ax[0].hist(snr_arr, bins=bins, histtype='step', density=True)
        ax[0].set_xlabel('SNR', fontsize=10)
        ax[0].set_ylabel('Relative Density', fontsize=10)

        ax[1].plot(*return_CDF(snr_arr))
        ax[1].set_xlim(5, 50)
        ax[1].set_xlabel('SNR', fontsize=10)
        ax[1].set_ylabel('CDF', fontsize=10)

        for lim in [25, 50, 75, 90]:
            snr = np.percentile(snr_arr, lim)
            ax[1].axvline(snr, color='r', alpha=.2, label='%i Perc: SNR %.1f' % (lim, snr))

        ax[1].legend()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)

        figures_dir = return_figures_dir()
        fname = f'{figures_dir}/field{field_id:06d}_obj_snr_cuts.png'
        fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
        print('-- %s saved' % fname)


if __name__ == '__main__':
    generate_obj_snr_cuts(*return_high_field_ids())
    generate_obj_snr_cuts(*return_low_field_ids())
