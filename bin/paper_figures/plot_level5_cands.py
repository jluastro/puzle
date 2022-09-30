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

from microlens.jlu.model import PSPL_Phot_Par_Param1

from puzle.models import CandidateLevel4
from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.jobs import return_num_objs_arr
from puzle.cands import return_best_obj
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
    num_objs_hist /= (1.5 * 1.5)
    num_objs_hist[num_objs_hist == 0] = 1

    fig, ax = plt.subplots(2, 2, figsize=(14, 7))
    ax = ax.flatten()
    for a in ax: a.clear()

    ax[0].set_title('ZTF Candidates Level 5', fontsize=14)
    ax[0].scatter(glon_cands_arr, glat_cands_arr, c='b', s=3)
    ax[0].set_xlim(-180, 180)
    ax[0].set_ylim(-90, 90)

    ax[1].set_title('ZTF Events Level 6', fontsize=14)
    ax[1].scatter(glon_cands_arr[clear_cond], glat_cands_arr[clear_cond], c='r', s=3)
    ax[1].set_xlim(-180, 180)
    ax[1].set_ylim(-90, 90)

    ax[2].set_title(r'Number of Objects with $N_{\rm epochs} \geq 20$', fontsize=14)
    norm = LogNorm(vmin=1e4, vmax=4e5)
    im = ax[2].imshow(num_objs_hist, norm=norm, extent=extent, origin='lower', aspect=0.75, cmap='viridis')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    # cbar = fig.colorbar(im, ax=ax[2])
    cbar.set_label(r'Number of Objects / deg$^2$', fontsize=12)
    ax[2].set_aspect('auto')

    ax[3].set_title('ZTF Events Level 6', fontsize=14)
    ax[3].scatter(glon_cands_arr[clear_cond], glat_cands_arr[clear_cond], c='r', s=3)
    ax[3].set_xlim(0, 150)
    ax[3].set_ylim(-30, 30)
    ax[3].axhline(-10, color='k', alpha=.2)
    ax[3].axhline(10, color='k', alpha=.2)

    for a in ax:
        a.plot(glon_ztf, glat_ztf, color='k')
        a.set_xlabel(r'l (degrees)')
        a.set_ylabel(r'b (degrees)')
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


def fetch_popsycle_tE_piE(glon_arr, glat_arr, seed=0, sample=True):
    np.random.seed(seed)
    popsycle_map_fname = return_data_dir() + '/popsycle_map.npz'
    popsycle_map = np.load(popsycle_map_fname)
    glon_popsycle = popsycle_map['l']
    glat_popsycle = popsycle_map['b']
    area_popsycle = popsycle_map['area']
    radius_popsycle = np.max(np.sqrt(area_popsycle / np.pi))

    popsycle_idx_dct = defaultdict(int)
    cond_arr = []
    for glon, glat in zip(glon_arr, glat_arr):
        dist = np.hypot(np.cos(np.radians(glat)) * (glon - glon_popsycle),
                        glat - glat_popsycle)
        if np.min(dist) < radius_popsycle + 6:
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
        popsycle_cond *= popsycle_catalog['delta_m_r'] >= 0.3
        popsycle_cond *= popsycle_catalog['ztf_r_app_LSN'] <= 21.5
        num_popsycle_cond = np.sum(popsycle_cond)
        print(idx, num, num_popsycle_cond)
        if num_popsycle_cond == 0:
            continue

        tE_popsycle_catalog = popsycle_catalog['t_E'][popsycle_cond]
        if sample:
            tE_popsycle_sample = np.random.choice(tE_popsycle_catalog, size=num, replace=True)
            tE_popsycle.extend(list(tE_popsycle_sample))
        else:
            tE_popsycle.extend(list(tE_popsycle_catalog))

        piE_popsycle_catalog = popsycle_catalog['pi_E'][popsycle_cond]
        if sample:
            piE_popsycle_sample = np.random.choice(piE_popsycle_catalog, size=num, replace=True)
            piE_popsycle.extend(list(piE_popsycle_sample))
        else:
            piE_popsycle.extend(list(piE_popsycle_catalog))

        BH_cond = popsycle_catalog['rem_id_L'] == 103
        if np.sum(BH_cond * popsycle_cond) != 0:
            tE_BH_popsycle.extend(list(popsycle_catalog['t_E'][popsycle_cond * BH_cond]))
            piE_BH_popsycle.extend(list(popsycle_catalog['pi_E'][popsycle_cond * BH_cond]))

    tE_popsycle = np.array(tE_popsycle)
    piE_popsycle = np.array(piE_popsycle)
    tE_BH_popsycle = np.array(tE_BH_popsycle)
    piE_BH_popsycle = np.array(piE_BH_popsycle)

    return cond_arr, tE_popsycle, piE_popsycle, tE_BH_popsycle, piE_BH_popsycle


def return_tE_ogle(max_num):
    tE_density_ogle = np.array([
        [5.5395015809930115, 0.07071284778943003],
        [7.55300129064756, 0.0824874655261459],
        [10.188854554294219, 0.14696843861124467],
        [14.041622573959856, 0.43867255910484987],
        [19.145475792949043, 0.3508734455902715],
        [26.10447912324336, 0.5969249509970737],
        [35.97551043496946, 1.015521127634569],
        [48.53027680430488, 0.9847160957933774],
        [66.17007649137457, 0.5832923594286206],
        [90.2215959024979, 0.3563193971228586],
        [123.01536886170413, 0.20309176209047378],
        [169.53185180413578, 0.2916640776023707],
        [228.69523172084783, 0.06447102107323877],
        [311.8214436979837, 0.04558929499981069],
        [429.73221371866487, 0.08441534596012994]])
    tE_ogle = tE_density_ogle[:, 0]
    tE_num_ogle = tE_density_ogle[:, 1] * max_num / np.max(tE_density_ogle[:, 1])
    return tE_ogle, tE_num_ogle


def plot_cands_tE_overlapping_popsycle():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        all()
    cands = np.array(cands)
    tE_cands = np.array([c.tE_pspl_gp for c in cands])
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    glon_arr = coords.galactic.l.value
    cond = glon_arr >= 180
    glon_arr[cond] -= 360
    glat_arr = coords.galactic.b.value

    cond_arr, tE_popsycle, _, _, _ = fetch_popsycle_tE_piE(glon_arr, glat_arr, seed=2, sample=False)
    tE_cands_overlap = np.array([c.tE_pspl_gp for c in cands[cond_arr]])

    bin_size = 11
    bins = np.logspace(np.log10(3), np.log10(500), bin_size)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    densityFlag = False
    tE_cands_num = np.histogram(tE_cands, bins=bins, density=densityFlag)[0]
    tE_cands_num_err = np.sqrt(tE_cands_num)
    tE_cands_overlap_num = np.histogram(tE_cands_overlap, bins=bins)[0]
    tE_cands_overlap_num_err = np.sqrt(tE_cands_overlap_num)

    tE_popsycle_counts = np.histogram(tE_popsycle, bins=bins, density=False)[0]
    factor = np.sum(tE_cands_overlap_num) / np.sum(tE_popsycle_counts)

    max_num = np.max(np.histogram(tE_popsycle, bins=bins)[0])
    tE_ogle, tE_num_ogle = return_tE_ogle(max_num * factor)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.clear()

    cond_non_zero = tE_cands_overlap_num != 0
    x = np.append(np.append(bin_centers[0], bin_centers[cond_non_zero]), 239.75)
    y = np.append(np.append(0.5, tE_cands_overlap_num[cond_non_zero]), 0.5)
    yerr = np.append(np.append(1, tE_cands_overlap_num_err[cond_non_zero]), 1)
    ax.errorbar(x, y, yerr=yerr,
                linestyle='None', color='m')
    ax.plot(x, y,
            linewidth=3, color='m', marker='.',
            label='ZTF Events Level 6: Galactic Plane')
    # ax.errorbar(bin_centers[cond_non_zero],
    #             tE_cands_overlap_num[cond_non_zero],
    #             yerr=tE_cands_overlap_num_err[cond_non_zero],
    #             linestyle='None', color='m')
    # ax.plot(bin_centers[cond_non_zero],
    #         tE_cands_overlap_num[cond_non_zero],
    #         linewidth=3, color='m', marker='.',
    #         label='ZTF Candidates Level 6: Galactic Plane')

    ax.hist(bins[:-1], bins=bins, weights=factor*tE_popsycle_counts, histtype='step', linewidth=2,
            color='b', label=r'Simulated $\mu$-Lens: Galactic Plane, scaled', density=densityFlag)
    ax.plot(tE_ogle, tE_num_ogle, color='k', marker='.', label='OGLE: Galactic Plane, scaled', alpha=.6)

    cond_non_zero = tE_cands_num != 0
    x = np.append(np.append(bin_centers[0], bin_centers[cond_non_zero]), 239.75)
    y = np.append(np.append(0.5, tE_cands_num[cond_non_zero]), 0.5)
    yerr = np.append(np.append(1, tE_cands_num_err[cond_non_zero]), 1)
    ax.errorbar(x, y, yerr=yerr,
                linestyle='None', color='g')
    ax.plot(x, y,
            linewidth=2, color='g', marker='.',
            label='ZTF Events Level 6: All Sky', linestyle='--')
    # ax.errorbar(bin_centers[cond_non_zero],
    #             tE_cands_num[cond_non_zero],
    #             yerr=tE_cands_num_err[cond_non_zero],
    #             linestyle='None', color='g')
    # ax.plot(bin_centers[cond_non_zero],
    #         tE_cands_num[cond_non_zero],
    #         linewidth=2, color='g', marker='.',
    #         label='ZTF Candidates Level 6: All Sky', linestyle='--')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t_E$ (days)')
    ax.set_ylabel('Number of Events')
    ax.set_ylim(5e-1, 5e1)
    ax.set_xlim(3, 500)
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[3], handles[1], handles[2]]
    labels = [labels[0], labels[3], labels[1], labels[2]]
    ax.legend(handles, labels, loc=8,
              markerscale=3, framealpha=1, fontsize=15)
    fig.tight_layout()

    fname = '%s/level5_cands_tE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.02)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_cands_tE_piE_overlapping_popsycle():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category != None).\
        all()
    cands = np.array(cands)
    ra_arr = np.array([c.ra for c in cands])
    dec_arr = np.array([c.dec for c in cands])
    category_arr = np.array([c.category for c in cands])
    clear_cond = category_arr == 'clear_microlensing'
    coords = SkyCoord(ra_arr, dec_arr, frame='icrs', unit='degree')
    glon_arr = coords.galactic.l.value
    cond = glon_arr >= 180
    glon_arr[cond] -= 360
    glat_arr = coords.galactic.b.value

    cond_arr, tE_popsycle, piE_popsycle, tE_BH_popsycle, piE_BH_popsycle = fetch_popsycle_tE_piE(glon_arr, glat_arr, sample=False)

    tE_cands = np.array([c.tE_pspl_gp for c in cands[cond_arr * clear_cond]])
    tE_err_cands = np.array([c.tE_err_pspl_gp for c in cands[cond_arr * clear_cond]])
    piE_cands = np.array([c.piE_pspl_gp for c in cands[cond_arr * clear_cond]])
    piE_err_cands = np.array([c.piE_err_pspl_gp for c in cands[cond_arr * clear_cond]])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()
    ax.errorbar(tE_cands, piE_cands,
                xerr=tE_err_cands, yerr=piE_err_cands,
                color='r', linestyle='', alpha=1)
    ax.scatter(tE_cands, piE_cands, color='r', s=5, label='ZTF Events Level 6')
    ax.scatter(tE_popsycle, piE_popsycle, color='b', s=5, label=r'Simulated $\mu$-Lens: Stellar Lens', alpha=.2)
    ax.scatter(tE_BH_popsycle, piE_BH_popsycle, color='k', s=5, label=r'Simulated $\mu$-Lens: BH Lens', alpha=.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$t_E$ (days)', fontsize=16)
    ax.set_ylabel(r'$\pi_E$', fontsize=16)
    ax.set_xlim(3e0, 3e3)
    ax.set_ylim(1e-2, 1e1)
    leg = ax.legend(markerscale=3, fontsize=14, loc=1)
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()

    fname = '%s/level5_cands_tE_piE.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.02)
    print('-- %s saved' % fname)
    plt.close(fig)


def _plot_lightcurve_axis(cand, ax, color_marker='k', color_model='r', remove_axis=True):
    obj = return_best_obj(cand)
    ax.scatter(obj.lightcurve.hmjd,
                     obj.lightcurve.mag, color=color_marker, s=3)

    pspl_gp_fit_dct = cand.pspl_gp_fit_dct
    source_id = cand.source_id_arr[cand.idx_best]
    color = cand.color_arr[cand.idx_best]
    model_params = pspl_gp_fit_dct[source_id][color]

    model = PSPL_Phot_Par_Param1(**model_params)
    hmjd_model = np.linspace(obj.lightcurve.hmjd.min(),
                             obj.lightcurve.hmjd.max(),
                             10000)
    mag_model = model.get_photometry(hmjd_model)
    ax.plot(hmjd_model, mag_model, color=color_model, alpha=.3)

    ax.invert_yaxis()
    if remove_axis:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        ax.tick_params(axis='both', labelsize=12)


def plot_level6_lightcurve_examples():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        order_by(CandidateLevel4.id).\
        all()
    print('%i total cands within the Galactic box' % len(cands))
    cands_plane = [c for c in cands if np.abs(c.glat) <= 15]
    print('%i cands within the Galactic plane' % len(cands_plane))
    cands_halo = [c for c in cands if np.abs(c.glat) > 15]
    print('%i cands within the Galactic plane' % len(cands_halo))

    # idx_plane = np.random.choice(np.arange(len(cands_plane)), replace=False, size=8)
    # idx_halo = np.random.choice(np.arange(len(cands_halo)), replace=False, size=8)
    idx_xlim_plane = [
        (77, [58600, 59250], [.03, .97], 'left'),
        (28, [58250, 58800], [.97, .97], 'right'),
        (23, [58250, 58500], [.03, .97], 'left'),
        (53, [58600, 59250], [.03, .97], 'left'),
        (10, [58250, 58800], [.97, .97], 'right'),
        (30, [58600, 59250], [.03, .97], 'left')
    ]
    idx_xlim_halo = [
        (5, [58250, 59250], [.03, .97], 'left'),
        (8, [58250, 59250], [.97, .97], 'right'),
        (12, [58400, 59250], [.03, .97], 'left'),
        (9, [58250, 59250], [.97, .97], 'right'),
        (34, [58250, 59250], [.97, .97], 'right'),
        (24, [58250, 59250], [.03, .97], 'left'),
    ]

    fig, ax = plt.subplots(4, 3, figsize=(12, 8))
    ax = ax.flatten()
    for j, (idx, xlim, textcoords, halign) in enumerate(idx_xlim_plane):
        ax[j].clear()
        cand = cands_plane[idx]
        # ax[j].set_title(f'P {idx} {cand.glat:.2f}', fontsize=12)
        _plot_lightcurve_axis(cand, ax[j], color_marker='m', color_model='k', remove_axis=False)
        ax[j].set_xlim(xlim)
        ax[j].text(textcoords[0], textcoords[1],
                   f'({cand.glon:.2f}, {cand.glat:.2f})',
                   horizontalalignment=halign, verticalalignment='top',
                   fontsize=12, transform=ax[j].transAxes)
    for j, (idx, xlim, textcoords, halign) in enumerate(idx_xlim_halo):
        ax[j+6].clear()
        cand = cands_halo[idx]
        # ax[j+6].set_title(f'H {idx} {cand.glat:.2f}', fontsize=12)
        _plot_lightcurve_axis(cand, ax[j+6], color_marker='g', color_model='k', remove_axis=False)
        ax[j+6].set_xlim(xlim)
        ax[j+6].text(textcoords[0], textcoords[1],
                     f'({cand.glon:.2f}, {cand.glat:.2f})',
                     horizontalalignment=halign, verticalalignment='top',
                     fontsize=12, transform=ax[j+6].transAxes)
    for a in ax:
        a.tick_params(axis='both', labelsize=12)
    #     if i == 3 and j == 1:
    #         ax[i, j].set_xlabel('heliocentric modified julian date (days)', fontsize=16)
    # if i == 2:
    #     ax[i, 0].set_ylabel('      magnitude', horizontalalignment='left', fontsize=16)
    fig.tight_layout(h_pad=1, w_pad=1)
    fig.subplots_adjust(bottom=.09, left=0.08)
    fig.text(0.5, 0.01, 'Heliocentric Modified Julian Date (days)',
             horizontalalignment='center',
             verticalalignment='bottom', fontsize=18)
    fig.text(0.01, 0.5, 'Magnitude',
             rotation='vertical',
             verticalalignment='center',
             horizontalalignment='left', fontsize=18)

    fname = '%s/level6_lightcurve_examples.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.05)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_lightcurve_examples():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category != None).\
        order_by(CandidateLevel4.id).\
        all()
    data = {'clear_microlensing':
                {'label': 'clear microlensing', 'idx_arr': [13, 63, 45]},
            'possible_microlensing':
                {'label': 'possible microlensing', 'idx_arr': [127, 300, 0]},
            'poor_model_data':
                {'label': 'poor model / data', 'idx_arr': [85, 41, 89]},
            'non_microlensing_variable':
                {'label': 'non-microlensing variable', 'idx_arr': [25, 18, 65]}
            }

    # 125, 127, 0, 300, 50

    # for i in range(10):
    fig, ax = plt.subplots(4, 3, figsize=(9, 8))
    for i, category in enumerate(data):
        cands_cat = [c for c in cands if c.category == category]
        label = data[category]['label']
        idx_arr = data[category]['idx_arr']
        # if category == 'possible_microlensing':
        #     idx_arr = np.random.choice(np.arange(len(cands_cat)),
        #                                size=3, replace=False)
        for j, idx in enumerate(idx_arr):
            cand = cands_cat[idx]
            ax[i, j].clear()
            if j == 1:
                ax[i, j].set_title(f'{label}', fontsize=18)
            _plot_lightcurve_axis(cand, ax[i, j], remove_axis=False)
            if i == 3 and j == 1:
                ax[i, j].set_xlabel('Heliocentric Modified Julian Date (days)', fontsize=16)
        if i == 2:
            ax[i, 0].set_ylabel('      Magnitude', horizontalalignment='left', fontsize=16)
    fig.tight_layout(w_pad=1)

    fname = '%s/level5_lightcurve_examples.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.05)
    print('-- %s saved' % fname)
    plt.close(fig)


def return_ztf_mag_cdf_3yr():
    ztf_mag_cdf_3yr = np.array([
        [19.00650314392299, 132.86047377045202]
        , [19.033409656317357, 137.78383741492735]
        , [19.060316168711722, 142.99124126966103]
        , [19.08722268110609, 148.10396505430867]
        , [19.114129193500457, 153.4060489791284]
        , [19.141035705894822, 159.18153325437834]
        , [19.16794221828919, 165.8091381604031]
        , [19.19484873068356, 172.3420629963416]
        , [19.221755243077926, 179.06434797245242]
        , [19.24866175547229, 185.21855252804653]
        , [19.27556826786666, 189.76319589217792]
        , [19.302474780261026, 193.7397588357926]
        , [19.32938129265539, 197.71632177940728]
        , [19.35628780504976, 201.69288472302242]
        , [19.383194317444126, 206.61624836749775]
        , [19.41010082983849, 213.33853334360833]
        , [19.43700734223286, 220.25017845989123]
        , [19.463913854627226, 227.0671435060881]
        , [19.490820367021595, 233.78942848219867]
        , [19.51772687941596, 240.41703338822322]
        , [19.54463339181033, 247.044638294248]
        , [19.571539904204695, 253.57756313018672]
        , [19.59844641659906, 260.1104879661252]
        , [19.62535292899343, 266.73809287215]
        , [19.652259441387795, 273.46037784826035]
        , [19.67916595378216, 280.27734289445743]
        , [19.70607246617653, 286.999627870568]
        , [19.732978978570895, 293.9112729868509]
        , [19.75988549096526, 300.63355796296173]
        , [19.78679200335963, 307.3558429390723]
        , [19.813698515753998, 314.0781279151829]
        , [19.840605028148364, 320.89509296138]
        , [19.86751154054273, 327.61737793749035]
        , [19.894418052937098, 334.33966291360116]
        , [19.921324565331464, 341.1566279597978]
        , [19.94823107772583, 348.0682730760809]
        , [19.975137590120198, 354.7905580521915]
        , [20.002044102514564, 362.3649636590769]
        , [20.02895061490893, 372.30637101811385]
        , [20.055857127303298, 382.43713851732286]
        , [20.082763639697664, 392.7572661567042]
        , [20.109670152092033, 403.0773937960853]
        , [20.136576664486398, 411.0305196833149]
        , [20.163483176880767, 416.5219637483067]
        , [20.190389689275133, 422.10808788338454]
        , [20.217296201669498, 427.50485187829054]
        , [20.244202714063867, 433.2803361535405]
        , [20.271109226458233, 442.08558267154467]
        , [20.298015738852598, 452.0269900305816]
        , [20.324922251246967, 461.96839738961853]
        , [20.351828763641333, 471.8151246785694]
        , [20.3787352760357, 481.4724918273482]
        , [20.405641788430067, 489.804337994922]
        , [20.432548300824436, 498.13618416249597]
        , [20.4594548132188, 506.4680303300697]
        , [20.486361325613167, 514.7051964275574]
        , [20.513267838007536, 523.2264027353035]
        , [20.5401743504019, 531.6529289729633]
        , [20.567080862796267, 540.0794552106231]
        , [20.595282341027797, 547.3976676866873]
        , [20.621383096901262, 561.2309828678506]
        , [20.647800399979367, 572.554719250144]
        , [20.674706912373736, 583.7269675202999]
        , [20.7016134247681, 594.9938958605417]
        , [20.72851993716247, 606.0714640606116]
        , [20.755426449556836, 616.3915916999927]
        , [20.782332961951205, 624.6287577974806]
        , [20.80923947434557, 632.4872036146239]
        , [20.836145986739936, 640.6296896420258]
        , [20.863052499134305, 648.5828155292552]
        , [20.88995901152867, 657.6721022575177]
        , [20.916865523923036, 667.5188295464686]
        , [20.943772036317405, 677.2708767653332]
        , [20.97067854871177, 687.2122841243702]
        , [20.99758506110614, 696.5856110628908]
        , [21.024491573500505, 704.0653365996902]
        , [21.051398085894874, 710.8823016458869]
        , [21.07830459828924, 717.8886268322558]
        , [21.105211110683605, 724.8949520186247]
        , [21.132117623077974, 732.6587177656822]
        , [21.15902413547234, 742.5054450546331]
        , [21.212837160261074, 760.3052982308136]
        , [21.23974367265544, 766.554182856494]
        , [21.266650185049805, 775.454109444584]
        , [21.293556697444174, 785.4901968737072]
        , [21.32046320983854, 795.6209643729164]
        , [21.34736972223291, 805.6570518020392]
        , [21.374276234627274, 816.1665395815928]
        , [21.401182747021643, 828.5696287628673]
        , [21.42808925941601, 841.2567581544001]
        , [21.454995771810374, 854.2279277561913]
        , [21.481902284204743, 867.0097372178102]
        , [21.50880879659911, 877.1405047170194]
        , [21.535715308993474, 884.0521498333021]
        , [21.562621821387843, 890.963794949585]
        , [21.58952833378221, 897.8754400658679]
        , [21.616434846176578, 904.5977250419787]
        , [21.643341358570943, 911.0359698078312]
        , [21.670247870965312, 917.2848544335116]
        , [21.697154383359678, 923.628419129278]
        , [21.724060895754043, 929.8773037549583]
        , [21.750967408148412, 936.6942688011551]
        , [21.777873920542778, 946.2569558798479]
        , [21.804780432937143, 955.9143230286265]
        , [21.831686945331512, 965.5716901774053]
        , [21.858593457725878, 975.1343772560981]
        , [21.885499970120243, 984.1289839142744]
        , [21.912406482514612, 992.2714699416761]
        , [21.939312994908978, 1000.4139559690777]
        , [21.966219507303347, 1008.5564419964794]
        , [21.993126019697712, 1016.1308476033647]])
    return ztf_mag_cdf_3yr


def return_ztf_mag_cdf_3yr_all_sky():
    ztf_mag_cdf_3yr_all_sky = np.array([
        [15.024339309556693, 137.02639685423878],
        [15.05124582195106, 141.09763986793973],
        [15.078152334345427, 145.45292309189904],
        [15.105058846739794, 149.99756645602997],
        [15.13196535913416, 154.6368898902474],
        [15.158871871528527, 160.22301402532503],
        [15.185778383922894, 166.56657872109167],
        [15.212684896317262, 172.9101434168581],
        [15.239591408711629, 179.25370811262474],
        [15.266497921105994, 185.21855252804653],
        [15.293404433500362, 189.76319589217792],
        [15.320310945894729, 193.9291189759649],
        [15.347217458289096, 197.9056819195796],
        [15.374123970683463, 201.88224486319427],
        [15.401030483077829, 206.90028857775587],
        [15.427936995472196, 213.52789348378042],
        [15.454843507866563, 220.43953860006354],
        [15.48175002026093, 227.3511837163462],
        [15.508656532655296, 234.2628288326291],
        [15.535563045049663, 240.60639352839553],
        [15.56246955744403, 246.1925176634736],
        [15.589376069838398, 251.4946015882931],
        [15.616282582232763, 257.27008586354304],
        [15.64318909462713, 263.1402502088795],
        [15.670095607021498, 270.8093358858507],
        [15.697002119415865, 278.85714184316635],
        [15.723908631810232, 286.5262275201378],
        [15.750815144204598, 294.38467333728136],
        [15.777721656598965, 302.3377992245107],
        [15.804628168993332, 310.19624504165427],
        [15.8315346813877, 317.7706506485397],
        [15.858441193782067, 325.62909646568323],
        [15.885347706176432, 333.48754228282655],
        [15.9122542185708, 342.48214894100283],
        [15.939160730965167, 352.0448360196956],
        [15.966067243359534, 361.32348288813023],
        [15.992973755753901, 370.886169966823],
        [16.019880268148267, 380.25949690534344],
        [16.046786780542636, 388.87538328317555],
        [16.073693292937, 397.2072294507493],
        [16.100599805331367, 405.72843575849515],
        [16.127506317725736, 414.0602819260689],
        [16.1544128301201, 421.25596725261016],
        [16.18131934251447, 426.93677145777406],
        [16.208225854908836, 432.712255733024],
        [16.2351323673032, 438.2036997980158],
        [16.26203887969757, 443.7898239330939],
        [16.288945392091936, 452.3110302408397],
        [16.3158519044863, 460.92691661867184],
        [16.34275841688067, 471.05768411788085],
        [16.369664929275036, 480.43101105640153],
        [16.396571441669405, 489.0468974342334],
        [16.42347795406377, 495.8638624804305],
        [16.450384466458136, 502.30210724628273],
        [16.477290978852505, 508.6456719420494],
        [16.50419749124687, 515.1785967779879],
        [16.53110400364124, 523.6051230156477],
        [16.558010516035605, 534.019930725115],
        [16.58491702842997, 544.6240985747545],
        [16.61398181721961, 555.4231959804533],
        [16.638730053218705, 567.8207157458407],
        [16.66563656561307, 578.9929640159967],
        [16.69254307800744, 589.4077717254638],
        [16.719449590401805, 599.8225794349312],
        [16.746356102796174, 610.2373871443986],
        [16.77326261519054, 619.8947542931771],
        [16.80016912758491, 627.8478801804067],
        [16.827075639979274, 635.4222857872921],
        [16.85398215237364, 643.3754116745217],
        [16.88088866476801, 651.0444973514932],
        [16.907795177162374, 659.0923033088086],
        [16.93470168955674, 667.7081896866407],
        [16.96160820195111, 676.3240760644728],
        [16.988514714345474, 684.8452823722187],
        [17.015421226739843, 693.0824484697064],
        [17.04232773913421, 698.8579327449565],
        [17.069234251528574, 703.9706565296042],
        [17.096140763922943, 709.36742052451],
        [17.12304727631731, 714.3854642390716],
        [17.149953788711677, 720.3503086544936],
        [17.176860301106043, 727.3566338408625],
        [17.20376681350041, 733.5108383964566],
        [17.230673325894777, 739.2863226717068],
        [17.257579838289143, 743.6416058956659],
        [17.284486350683512, 749.3224101008298],
        [17.311392863077877, 755.8553349367685],
        [17.338299375472243, 762.4829398427933],
        [17.365205887866612, 768.9211846086455],
        [17.392112400260977, 776.4955902155309],
        [17.419018912655346, 789.3720797472361],
        [17.445925425049712, 802.8166496994572],
        [17.472831937444077, 816.639939932023],
        [17.499738449838446, 830.0845098842444],
        [17.52664496223281, 840.5939976637978],
        [17.553551474627177, 845.8014015185315],
        [17.580457987021546, 851.0088053732652],
        [17.60736449941591, 855.9321690177405],
        [17.63427101181028, 861.139572872474],
        [17.661177524204646, 865.7788963066914],
        [17.688084036599015, 870.7022599511668],
        [17.71499054899338, 875.5309435255563],
        [17.741897061387746, 880.3596270999456],
        [17.768803573782115, 885.9457512350236],
        [17.79571008617648, 895.5084383137162],
        [17.822616598570846, 905.0711253924089],
        [17.849523110965215, 914.9178526813598],
        [17.87642962335958, 924.7645799703107],
        [17.90333613575395, 932.9070659977125],
        [17.930242648148315, 939.5346709037372],
        [17.95714916054268, 945.8782355995036],
        [17.98405567293705, 952.2218002952701],
        [18.010962185331415, 958.4706849209505]])
    return ztf_mag_cdf_3yr_all_sky


def return_ztf_mag_cdf_1yr():
    ztf_mag_cdf_1yr = np.array([
        [19.00650314392299, 43.29312746903315],
        [19.033409656317357, 45.28140894084072],
        [19.060316168711722, 47.26969041264829],
        [19.08722268110609, 49.35265195454167],
        [19.114129193500457, 51.24625335626297],
        [19.141035705894822, 53.423894968242394],
        [19.16794221828919, 55.980256860566215],
        [19.19484873068356, 58.44193868280399],
        [19.221755243077926, 60.903620505041545],
        [19.24866175547229, 63.36530232727932],
        [19.27556826786666, 64.02806281788185],
        [19.302474780261026, 65.54294393925898],
        [19.32938129265539, 66.11102435977523],
        [19.35628780504976, 67.0578250606361],
        [19.383194317444126, 68.47802611192697],
        [19.41010082983849, 70.93970793416474],
        [19.43700734223286, 73.4013897564023],
        [19.463913854627226, 75.67371143846799],
        [19.490820367021595, 78.13539326070577],
        [19.51772687941596, 80.21835480259915],
        [19.54463339181033, 82.30131634449253],
        [19.571539904204695, 84.19491774621383],
        [19.59844641659906, 86.27787928810721],
        [19.62535292899343, 88.17148068982874],
        [19.652259441387795, 89.02360132060335],
        [19.67916595378216, 90.44380237189421],
        [19.70607246617653, 91.10656286249673],
        [19.732978978570895, 92.43208384370178],
        [19.75988549096526, 93.85228489499286],
        [19.78679200335963, 95.93524643688625],
        [19.813698515753998, 98.01820797877986],
        [19.840849632806492, 100.04246787721968],
        [19.86751154054273, 102.3734912027387],
        [19.894418052937098, 104.83517302497648],
        [19.921324565331464, 107.3915349173003],
        [19.94823107772583, 110.3266170899683],
        [19.975137590120198, 112.97765905237816],
        [20.002044102514564, 115.72338108487406],
        [20.02895061490893, 119.51058388831666],
        [20.055857127303298, 123.39246676184553],
        [20.082763639697664, 127.27434963537416],
        [20.109670152092033, 131.06155243881676],
        [20.136576664486398, 133.99663461148498],
        [20.163483176880767, 136.07959615337836],
        [20.190389689275133, 137.97319755509966],
        [20.217296201669498, 140.05615909699304],
        [20.244202714063867, 141.94976049871434],
        [20.271109226458233, 144.03272204060772],
        [20.298015738852598, 146.11568358250133],
        [20.324922251246967, 148.19864512439472],
        [20.351828763641333, 150.09224652611624],
        [20.3787352760357, 152.5539283483538],
        [20.405641788430067, 156.62517136205474],
        [20.432548300824436, 160.60173430566942],
        [20.4594548132188, 164.76765738945642],
        [20.486361325613167, 168.7442203330711],
        [20.513267838007536, 171.8686626459114],
        [20.5401743504019, 174.2356643980629],
        [20.567080862796267, 176.79202629038673],
        [20.595282341027797, 176.14040463155925],
        [20.620893887585, 184.36643189727215],
        [20.647800399979367, 189.19511547166167],
        [20.674706912373736, 194.21315918622304],
        [20.7016134247681, 199.2312029007844],
        [20.72851993716247, 203.96520640508766],
        [20.755426449556836, 208.32048962904696],
        [20.782332961951205, 210.87685152137078],
        [20.80923947434557, 213.52789348378042],
        [20.836145986739936, 216.27361551627655],
        [20.863052499134305, 219.01933754877223],
        [20.88995901152867, 220.8182588804077],
        [20.916865523923036, 222.42782007187066],
        [20.943772036317405, 224.32142147359218],
        [20.97067854871177, 226.02566273514117],
        [20.99758506110614, 227.5405438565183],
        [21.024491573500505, 230.00222567875608],
        [21.051398085894874, 232.74794771125198],
        [21.07830459828924, 235.49366974374811],
        [21.105211110683605, 237.95535156598567],
        [21.132117623077974, 240.60639352839553],
        [21.15902413547234, 242.97339528054704],
        [21.185930647866705, 246.28719773355942],
        [21.212837160261074, 247.51803864467843],
        [21.23974367265544, 249.41164004639973],
        [21.266650185049805, 251.39992151820707],
        [21.293556697444174, 253.29352291992836],
        [21.32046320983854, 255.18712432164966],
        [21.34736972223291, 257.08072572337096],
        [21.374276234627274, 258.9743271250925],
        [21.401182747021643, 261.43600894733004],
        [21.42808925941601, 264.18173097982617],
        [21.454995771810374, 266.6434128020637],
        [21.481902284204743, 269.01041455421546],
        [21.50880879659911, 271.2827362362809],
        [21.535715308993474, 272.98697749783037],
        [21.562621821387843, 274.69121875937935],
        [21.58952833378221, 276.4901400910146],
        [21.616434846176578, 278.4784215628222],
        [21.643341358570943, 280.3720229645435],
        [21.670247870965312, 282.64434464660917],
        [21.697154383359678, 284.8219862585886],
        [21.724060895754043, 286.999627870568],
        [21.750967408148412, 289.6506698329779],
        [21.777873920542778, 293.34319256633444],
        [21.804780432937143, 297.31975550994935],
        [21.831686945331512, 301.201638383478],
        [21.858593457725878, 305.27288139717894],
        [21.885499970120243, 308.7760439903632],
        [21.912406482514612, 312.3738866536339],
        [21.939312994908978, 315.498328966474],
        [21.966219507303347, 319.00149155965846],
        [21.98334183337249, 320.57949272775954]])
    return ztf_mag_cdf_1yr


def plot_cands_magnitude():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        all()
    cands_plane_box = [c for c in cands if c.glon >= 10 and c.glon <= 100 and c.glat >= -10 and c.glat <= 10]
    print('%i cands within the Galactic box' % len(cands_plane_box))

    def return_mag_base_arrs(cands):
        mag_base_arrs = {
            'g': [],
            'r': [],
            'i': []
        }

        for cand in cands:
            cand_fitter_data = load_cand_fitter_data(cand.id)
            filters = [p.split('_')[-1] for p in cand_fitter_data['data']['phot_files']]
            assert len(cand.mag_base_arr_pspl_gp[:3]) == len(filters)
            for filt in mag_base_arrs:
                mag_bases = [m for i, m in enumerate(cand.mag_base_arr_pspl_gp[:3]) if filters[i] == filt]
                if len(mag_bases) > 0:
                    mag_base = np.median(mag_bases)
                    mag_base_arrs[filt].append(mag_base)
        return mag_base_arrs

    mag_base_arrs = return_mag_base_arrs(cands)
    for filter in mag_base_arrs:
        print('%i candidates with %s baseline' % (len(mag_base_arrs[filter]), filter))
    mag_base_plane_box_arrs = return_mag_base_arrs(cands_plane_box)

    def return_CDF(arr):
        x = np.sort(arr)
        y = np.arange(len(arr))
        return x, y

    color_r_band = '#e93574'
    color_g_band = '#47aaae'
    color_fit = '#564787'

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    for a in ax: a.clear()
    bins = np.linspace(14, 21.5, 15)
    ax[0].set_title(r'ZTF Events Level 6', fontsize=16)
    ax[0].hist(mag_base_arrs['r'], histtype='step', color=color_r_band, bins=bins, label='r-band', linewidth=2)
    ax[0].hist(mag_base_arrs['g'], histtype='step', color=color_g_band, bins=bins, label='g-band', linewidth=2)
    ax[0].hist(mag_base_arrs['i'], histtype='step', color='k', bins=bins, label='i-band', linewidth=2)
    ax[0].set_xlabel('Magnitude')
    ax[0].set_ylabel('Number of Events')
    ax[0].legend(loc=2, fontsize=14)

    ax[1].plot(*return_CDF(mag_base_arrs['r']), color=color_r_band, linestyle='-',
               label=r'r-band: All Sky', linewidth=2)
    ax[1].plot(*return_CDF(mag_base_plane_box_arrs['r']), color=color_r_band, linestyle='--',
               label=r'r-band: Medford2020 Search Area', linewidth=2)

    ztf_mag_cdf_3yr = return_ztf_mag_cdf_3yr()
    ax[1].plot(*ztf_mag_cdf_3yr.T, color=color_fit, linestyle='-', label='Medford2020 Simulation')
    m, b = np.polyfit(*np.log10(ztf_mag_cdf_3yr[:60]).T, deg=1)
    x = np.linspace(14, 19, 1000)
    y = 10**(m*np.log10(x)+b-.05)
    ax[1].plot(x, y, color=color_fit, linestyle='--', label='Medford2020 Simulation, extended')

    for a in ax:
        a.set_xlim(13.9, 21.6)

    ax[1].set_ylim(1, 1000)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Faintest Magnitude')
    ax[1].set_ylabel('Total Number of Events')
    ax[1].legend(loc=2, fontsize=13)

    fig.tight_layout()

    fname = '%s/level6_magnitudes.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.05)
    print('-- %s saved' % fname)
    plt.close(fig)


def plot_cands_blend_fraction():
    cands = CandidateLevel4.query.\
        filter(CandidateLevel4.level5 == True,
               CandidateLevel4.category == 'clear_microlensing').\
        all()

    def return_b_sff_arrs(cands):
        b_sff_arrs = {
            'g': [],
            'r': [],
            'i': []
        }

        for cand in cands:
            cand_fitter_data = load_cand_fitter_data(cand.id)
            filters = [p.split('_')[-1] for p in cand_fitter_data['data']['phot_files']]
            assert len(cand.b_sff_arr_pspl_gp[:3]) == len(filters)
            for filt in b_sff_arrs:
                b_sffs = [m for i, m in enumerate(cand.b_sff_arr_pspl_gp[:3]) if filters[i] == filt]
                if len(b_sffs) > 0:
                    b_sff = np.median(b_sffs)
                    b_sff_arrs[filt].append(b_sff)
        return b_sff_arrs

    b_sff_arrs = return_b_sff_arrs(cands)

    color_r_band = '#e93574'
    color_g_band = '#47aaae'

    bins = np.linspace(0, 1.5, 12)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.clear()

    ax.set_title(r'ZTF Events Level 6', fontsize=16)
    ax.hist(b_sff_arrs['r'], histtype='step', color=color_r_band, bins=bins, label='r-band', linewidth=2)
    ax.hist(b_sff_arrs['g'], histtype='step', color=color_g_band, bins=bins, label='g-band', linewidth=2)
    ax.hist(b_sff_arrs['i'], histtype='step', color='k', bins=bins, label='i-band', linewidth=2)
    ax.set_xlabel('Source Flux Fraction')
    ax.set_ylabel('Number of Events')
    ax.legend(loc=2, fontsize=14)

    fig.tight_layout()

    fname = '%s/level6_source_flux_fractions.png' % return_figures_dir()
    fig.savefig(fname, dpi=100, bbox_inches='tight', pad_inches=0.05)
    print('-- %s saved' % fname)
    plt.close(fig)


def generate_all_figures():
    plot_cands_on_sky()
    plot_cands_tE_overlapping_popsycle()
    plot_cands_tE_piE_overlapping_popsycle()
    #plot_lightcurve_examples()
    #plot_level6_lightcurve_examples()
    plot_cands_magnitude()
    plot_cands_blend_fraction()


if __name__ == '__main__':
    generate_all_figures()
