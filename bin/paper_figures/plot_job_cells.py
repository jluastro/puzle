#! /usr/bin/env python
"""
plot_job_cells.py
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.coordinates import SkyCoord

from puzle.utils import return_figures_dir
from puzle.jobs import return_num_objs_arr


def plot_job_cells():
    ra_arr, dec_arr, num_objs_arr = return_num_objs_arr()

    ra_bins = np.arange(0, 361, 2)
    dec_bins = np.arange(-30, 91, 1)
    extent = (ra_bins.min(), ra_bins.max(), dec_bins.min(), dec_bins.max())
    job_hist = np.histogram2d(ra_arr, dec_arr, bins=(ra_bins, dec_bins))[0].T
    num_objs_hist = binned_statistic_2d(ra_arr, dec_arr, num_objs_arr,
                                        statistic='sum', bins=(ra_bins, dec_bins))[0].T

    glon = np.linspace(0, 360, 10000)
    
    glat_low = np.zeros(len(glon)) - 20
    coords_low = SkyCoord(glon, glat_low, frame='galactic', unit='degree')
    ra_gal_low = coords_low.icrs.ra.value
    dec_gal_low = coords_low.icrs.dec.value

    glat_high = np.zeros(len(glon)) + 20
    coords_high = SkyCoord(glon, glat_high, frame='galactic', unit='degree')
    ra_gal_high = coords_high.icrs.ra.value
    dec_gal_high = coords_high.icrs.dec.value

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for a in ax: a.clear()

    ax[0].set_title('Star Process Jobs')
    im0 = ax[0].imshow(job_hist / 2, extent=extent, origin='lower', cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=ax[0])
    cbar0.set_label(r'Number of Jobs / deg$^2$')

    ax[1].set_title(r'Objects with $n_{\rm epochs} \geq 20$')
    norm = LogNorm(vmin=1e4, vmax=1e6)
    im0 = ax[1].imshow(num_objs_hist / 2, norm=norm, extent=extent, origin='lower', cmap='viridis')
    cbar0 = fig.colorbar(im0, ax=ax[1])
    cbar0.set_label(r'Number of Objects / deg$^2$')

    for a in ax:
        a.scatter(ra_gal_low, dec_gal_low, c='k', s=.1, alpha=.05)
        a.scatter(ra_gal_high, dec_gal_high, c='k', s=.1, alpha=.05)
        a.set_xlabel('right ascension (degrees)')
        a.set_ylabel('declination (degrees)')
        a.set_xlim(0, 360)
        a.set_ylim(-28, 90)
    fig.tight_layout(h_pad=1)

    fname = '%s/job_cells.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


if __name__ == '__main__':
    plot_job_cells()
