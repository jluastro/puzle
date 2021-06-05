#! /usr/bin/env python
"""
plot_level5_cands.py
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import LogNorm
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from collections import defaultdict
from scipy.stats import binned_statistic_2d

from puzle.models import CandidateLevel4
from puzle.jobs import return_num_objs_arr
from puzle.utils import return_figures_dir, return_data_dir


def plot_cands_on_sky():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category != None).\
        with_entities(CandidateLevel4.ra, CandidateLevel4.dec,
                      CandidateLevel4.category).\
        all()
    ra_cands_arr = np.array([c.ra for c in cands])
    dec_cands_arr = np.array([c.dec for c in cands])
    category_arr = np.array([c.category for c in cands])
    clear_cond = category_arr == 'clear_microlensing'
    coords_cands = SkyCoord(ra_cands_arr, dec_cands_arr, frame='icrs', unit='degree')
    glon_cands_arr = coords_cands.galactic.l.value
    lb_cond = glon_cands_arr >= 180
    glon_cands_arr[lb_cond] -= 360
    glat_cands_arr = coords_cands.galactic.b.value

    ra_ztf = np.linspace(0, 360, 10000)
    dec_ztf = np.ones(len(ra_ztf)) * -30
    coords_ztf = SkyCoord(ra_ztf, dec_ztf, frame='icrs', unit='degree')
    glon_ztf = coords_ztf.galactic.l.value
    lb_cond = glon_ztf >= 180
    glon_ztf[lb_cond] -= 360
    glat_ztf = coords_ztf.galactic.b.value

    ra_objs_arr, dec_objs_arr, num_objs_arr = return_num_objs_arr()
    coords_objs = SkyCoord(ra_objs_arr, dec_objs_arr, frame='icrs', unit='degree')
    glon_objs_arr = coords_objs.galactic.l.value
    lb_cond = glon_objs_arr >= 180
    glon_objs_arr[lb_cond] -= 360
    glat_objs_arr = coords_objs.galactic.b.value
    glon_bins = np.arange(-180, 181, 1.5)
    glat_bins = np.arange(-90, 90, 1.5)
    extent = (glon_bins.min(), glon_bins.max(), glat_bins.min(), glat_bins.max())
    num_objs_hist = binned_statistic_2d(glon_objs_arr, glat_objs_arr, num_objs_arr,
                                        statistic='sum', bins=(glon_bins, glat_bins))[0].T
    num_objs_hist[num_objs_hist == 0] = 1

    fig, ax = plt.subplots(2, 2, figsize=(14, 6))
    ax = ax.flatten()
    for a in ax: a.clear()

    ax[0].set_title('Level 5 Candidates')
    ax[0].scatter(glon_cands_arr, glat_cands_arr, c='b', s=3)
    ax[0].set_xlim(-180, 180)
    ax[0].set_ylim(-90, 90)

    ax[1].set_title('Level 6 Candidates')
    ax[1].scatter(glon_cands_arr[clear_cond], glat_cands_arr[clear_cond], c='r', s=3)
    ax[1].set_xlim(-180, 180)
    ax[1].set_ylim(-90, 90)

    ax[2].set_title(r'Number of Objects with $N_{\rm epochs} \geq 20$')
    norm = LogNorm(vmin=1e4, vmax=8e5)
    ax[2].imshow(num_objs_hist, norm=norm, extent=extent, origin='lower', aspect=0.75)

    ax[3].set_title('Level 6 Candidates - Galactic Plane')
    ax[3].scatter(glon_cands_arr[clear_cond], glat_cands_arr[clear_cond], c='r', s=3)
    ax[3].set_xlim(0, 150)
    ax[3].set_ylim(-30, 30)
    ax[3].axhline(-10, color='k', alpha=.2)
    ax[3].axhline(10, color='k', alpha=.2)

    for a in ax:
        a.plot(glon_ztf, glat_ztf, color='k')
        a.set_xlabel('glon', fontsize=16)
        a.set_ylabel('glat', fontsize=16)
        if a == ax[-1]:
            continue
        rect = Rectangle((0, -30), 150, 60, facecolor='none', edgecolor='k')
        a.add_patch(rect)

    fig.tight_layout()

    for xy in [(0, -30), (150, -30), (150, 30), (0, 30)]:
        transFigure = fig.transFigure.inverted()
        coord1 = transFigure.transform(ax[1].transData.transform([xy[0], xy[1]]))
        coord2 = transFigure.transform(ax[3].transData.transform([xy[0], xy[1]]))
        line = matplotlib.lines.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]),
                                       transform=fig.transFigure, color='k', alpha=.5, linewidth=.5)
        fig.add_artist(line)

    popsycle_map_fname = return_data_dir() + '/popsycle_map.npz'
    popsycle_map = np.load(popsycle_map_fname)
    for i in range(len(popsycle_map['l'])):
        l = popsycle_map['l'][i]
        b = popsycle_map['b'][i]
        area = popsycle_map['area'][i]
        radius = np.sqrt(area / np.pi)
        for a in ax:
            if a == ax[2]:
                continue
            patch = Circle((l, b), radius=radius+6,
                           facecolor='g', edgecolor='None', alpha=0.2)
            a.add_patch(patch)

    plane_cond = np.abs(glat_cands_arr) <= 10
    num_clear_cand = np.sum(clear_cond)
    num_clear_plane_cand = np.sum(clear_cond * plane_cond)
    frac_clear_plane_cand = 100 * num_clear_plane_cand / num_clear_cand
    print(f'{num_clear_cand} clear microlensing candidates')
    print(f'{num_clear_plane_cand} clear microlensing candidates within galactic plane')
    print(f'{frac_clear_plane_cand:.1f}% of clear microlensing candidates within galactic plane')

    fname = '%s/level5_cands_on_sky.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def fetch_popsycle_tE_piE(glon_arr, glat_arr, seed=0):
    np.random.seed(seed)
    popsycle_map_fname = return_data_dir() + '/popsycle_map.npz'
    popsycle_map = np.load(popsycle_map_fname)
    glon_popsycle = popsycle_map['l']
    glat_popsycle = popsycle_map['b']

    popsycle_idx_dct = defaultdict(int)
    cond_arr = []
    for glon, glat in zip(glon_arr, glat_arr):
        dist = np.hypot(np.cos(np.radians(glat)) * (glon - glon_popsycle),
                        glat - glat_popsycle)
        if np.min(dist) < 6:
            popsycle_idx_dct[np.argmin(dist)] += 1
            cond_arr.append(True)
        else:
            cond_arr.append(False)

    popsycle_base_folder = '/global/cfs/cdirs/uLens/PopSyCLE_runs/PopSyCLE_runs_v3_refined_events'
    tE_popsycle = []
    piE_popsycle = []
    tE_BH_popsycle = []
    piE_BH_popsycle = []
    for idx, num in popsycle_idx_dct.items():
        l = glon_popsycle[idx]
        b = glat_popsycle[idx]
        popsycle_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
        popsycle_catalog = Table.read(popsycle_fname, format='fits')
        popsycle_cond = popsycle_catalog['u0'] <= 1.0
        popsycle_cond *= popsycle_catalog['delta_m_r'] >= 0.1
        popsycle_cond *= popsycle_catalog['ztf_r_app_LSN'] <= 22
        num_popsycle_cond = np.sum(popsycle_cond)
        if num_popsycle_cond == 0:
            continue

        tE_popsycle_catalog = popsycle_catalog['t_E'][popsycle_cond]
        tE_popsycle_sample = np.random.choice(tE_popsycle_catalog, size=num, replace=True)
        tE_popsycle.extend(list(tE_popsycle_sample))

        piE_popsycle_catalog = popsycle_catalog['pi_E'][popsycle_cond]
        piE_popsycle_sample = np.random.choice(piE_popsycle_catalog, size=num, replace=True)
        piE_popsycle.extend(list(piE_popsycle_sample))

        BH_cond = popsycle_catalog['rem_id_L'] == 103
        if np.sum(BH_cond * popsycle_cond) != 0:
            tE_BH_popsycle.extend(list(popsycle_catalog['t_E'][popsycle_cond * BH_cond]))
            piE_BH_popsycle.extend(list(popsycle_catalog['pi_E'][popsycle_cond * BH_cond]))

    tE_popsycle = np.array(tE_popsycle)
    piE_popsycle = np.array(piE_popsycle)
    tE_BH_popsycle = np.array(tE_BH_popsycle)
    piE_BH_popsycle = np.array(piE_BH_popsycle)

    return cond_arr, tE_popsycle, piE_popsycle, tE_BH_popsycle, piE_BH_popsycle


def plot_cands_tE_overlapping_popsycle():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        all()
    cands = np.array(cands)
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    glon_arr = coords.galactic.l.value
    cond = glon_arr >= 180
    glon_arr[cond] -= 360
    glat_arr = coords.galactic.b.value

    cond_arr, tE_popsycle, _, _, _ = fetch_popsycle_tE_piE(glon_arr, glat_arr)
    tE_cands = np.array([c.tE_pspl_gp for c in cands[cond_arr]])

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(np.log10(np.min([tE_cands.min(), tE_popsycle.min()])),
                       np.log10(np.max([tE_cands.max(), tE_popsycle.max()])),
                       20)
    ax.clear()
    ax.hist(tE_cands, bins=bins, histtype='step', color='r', label='candidates')
    ax.hist(tE_popsycle, bins=bins, histtype='step', color='b', label='popsycle')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t_E$', fontsize=16)
    ax.legend()
    fig.tight_layout()

    fname = '%s/level5_cands_tE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_cands_tE_piE_overlapping_popsycle():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        all()
    cands = np.array(cands)
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    glon_arr = coords.galactic.l.value
    cond = glon_arr >= 180
    glon_arr[cond] -= 360
    glat_arr = coords.galactic.b.value

    cond_arr, tE_popsycle, piE_popsycle, tE_BH_popsycle, piE_BH_popsycle = fetch_popsycle_tE_piE(glon_arr, glat_arr)

    tE_cands = np.array([c.tE_pspl_gp for c in cands[cond_arr]])
    tE_err_cands = np.array([c.tE_err_pspl_gp for c in cands[cond_arr]])
    piE_cands = np.array([c.piE_pspl_gp for c in cands[cond_arr]])
    piE_err_cands = np.array([c.piE_err_pspl_gp for c in cands[cond_arr]])
    cond_piE_err = piE_err_cands <= 0.1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    ax.errorbar(tE_cands[~cond_piE_err], piE_cands[~cond_piE_err],
                xerr=tE_err_cands[~cond_piE_err], yerr=piE_err_cands[~cond_piE_err],
                color='r', linestyle='', alpha=.2)
    ax.scatter(tE_cands[~cond_piE_err], piE_cands[~cond_piE_err], color='r', s=5, label='candidates')
    ax.errorbar(tE_cands[cond_piE_err], piE_cands[cond_piE_err],
                xerr=tE_err_cands[cond_piE_err], yerr=piE_err_cands[cond_piE_err],
                color='g', linestyle='', alpha=.2)
    ax.scatter(tE_cands[cond_piE_err], piE_cands[cond_piE_err], color='g', s=5, label='candidates cut on u0_err, piE_err')
    ax.scatter(tE_popsycle, piE_popsycle, color='b', s=5, label='sample PopSyCLE stars + BH')
    ax.scatter(tE_BH_popsycle, piE_BH_popsycle, color='c', s=5, label='all PopSyCLE BH')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t_E$', fontsize=16)
    ax.set_ylabel(r'$\pi_E$', fontsize=16)
    ax.legend(markerscale=3)
    fig.tight_layout()

    fname = '%s/level5_cands_tE_piE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_figures():
    plot_cands_on_sky()
    plot_cands_tE_overlapping_popsycle()
    plot_cands_tE_piE_overlapping_popsycle()


if __name__ == '__main__':
    generate_all_figures()
