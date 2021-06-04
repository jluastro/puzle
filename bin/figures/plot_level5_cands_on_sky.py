#! /usr/bin/env python
"""
plot_level5_cands_on_sky.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from collections import defaultdict

from puzle.models import CandidateLevel4
from puzle.utils import return_figures_dir, return_data_dir


def plot_cands_on_sky():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        with_entities(CandidateLevel4.ra, CandidateLevel4.dec).\
        all()
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    glon_arr = coords.galactic.l.value
    cond = glon_arr >= 180
    glon_arr[cond] -= 360
    glat_arr = coords.galactic.b.value

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    ax.scatter(glon_arr, glat_arr, c='b', s=5)
    ax.set_xlabel('glon', fontsize=16)
    ax.set_ylabel('glat', fontsize=16)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.tight_layout()

    popsycle_map_fname = return_data_dir() + '/popsycle_map.npz'
    popsycle_map = np.load(popsycle_map_fname)
    for i in range(len(popsycle_map['l'])):
        l = popsycle_map['l'][i]
        b = popsycle_map['b'][i]
        area = popsycle_map['area'][i]
        radius = np.sqrt(area / np.pi)
        patch = Circle((l, b), radius=radius,
                       facecolor='r', edgecolor='None', alpha=0.4)
        ax.add_patch(patch)
        patch = Circle((l, b), radius=radius+6,
                       facecolor='g', edgecolor='None', alpha=0.2)
        ax.add_patch(patch)

    fname = '%s/level5_cands_on_sky.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


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
    tE_cands = [c.tE_pspl_gp for c in cands[cond_arr]]
    tE_popsycle = []
    for idx, num in popsycle_idx_dct.items():
        popsycle_fname = f'{popsycle_base_folder}/l{l:.1f}_b{b:.1f}_refined_events_ztf_r_Damineli16.fits'
        popsycle_catalog = Table.read(popsycle_fname, format='fits')
        popsycle_cond = popsycle_catalog['u0'] <= 1.0
        popsycle_cond *= popsycle_catalog['delta_m_r'] >= 0.1
        popsycle_cond *= popsycle_catalog['f_blend_r'] >= 0.1
        popsycle_cond *= popsycle_catalog['ztf_r_app_LSN'] <= 22
        tE_popsycle_catalog = popsycle_catalog['t_E'][popsycle_cond]
        tE_popsycle_sample = np.random.choice(tE_popsycle_catalog, size=num, replace=True)
        tE_popsycle.extend(list(tE_popsycle_sample))

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(0, 3, 20)
    ax.clear()
    ax.hist(tE_cands, bins=bins, histtype='step', color='r')
    ax.hist(tE_popsycle, bins=bins, histtype='step', color='b')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()

    fname = '%s/level5_cands_tE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.01)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_figures():
    plot_cands_on_sky()


if __name__ == '__main__':
    generate_all_figures()
