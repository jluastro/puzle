#! /usr/bin/env python
"""
export_candidates.py
"""

import numpy as np
from astropy.io import fits

from puzle.pspl_gp_fit import load_cand_fitter_data
from puzle.utils import MJD_finish, return_data_dir
from puzle.models import CandidateLevel3, CandidateLevel4
from puzle import db


def return_colnames():
    colnames = "id,ra,dec,t0,t0_err," \
               "tE,tE_err,u0_amp,u0_amp_err," \
               "piE_E,piE_E_err,piE_N,piE_N_err," \
               "mag_base_r,mag_base_err_r," \
               "mag_base_g,mag_base_err_g," \
               "mag_base_i,mag_base_err_i," \
               "b_sff_r,b_sff_err_r," \
               "b_sff_g,b_sff_err_g," \
               "b_sff_i,b_sff_err_i"
    return colnames.split(',')


def construct_arrays(cands):
    colnames = return_colnames()
    arrays = {}
    for colname in colnames:
        if 'mag' in colname or 'sff' in colname:
            array = []
        else:
            if colname in ['id', 'ra', 'dec']:
                suffix = ''
            else:
                suffix = '_pspl_gp'
            cand_colname = '%s%s' % (colname, suffix)
            if colname == 'id':
                array = ['PUZLE_%s' % getattr(c, cand_colname) for c in cands]
            else:
                array = [getattr(c, cand_colname) for c in cands]
        arrays[colname] = array

    for cand in cands:
        cand_fitter_data = load_cand_fitter_data(cand.id)
        filters = [p.split('_')[-1] for p in cand_fitter_data['data']['phot_files']]
        assert len(cand.b_sff_arr_pspl_gp[:3]) == len(filters)
        for filt in ['g', 'r', 'i']:
            b_sff_arr = [m for i, m in enumerate(cand.b_sff_arr_pspl_gp[:3]) if filters[i] == filt]
            b_sff_err_arr = [m for i, m in enumerate(cand.b_sff_err_arr_pspl_gp[:3]) if filters[i] == filt]
            mag_base_arr = [m for i, m in enumerate(cand.mag_base_arr_pspl_gp[:3]) if filters[i] == filt]
            mag_base_err_arr = [m for i, m in enumerate(cand.mag_base_err_arr_pspl_gp[:3]) if filters[i] == filt]
            if len(b_sff_arr) > 0:
                arrays['b_sff_%s' % filt].append(np.median(b_sff_arr))
                arrays['b_sff_err_%s' % filt].append(np.median(b_sff_err_arr))
                arrays['mag_base_%s' % filt].append(np.median(mag_base_arr))
                arrays['mag_base_err_%s' % filt].append(np.median(mag_base_err_arr))
            else:
                arrays['b_sff_%s' % filt].append(None)
                arrays['b_sff_err_%s' % filt].append(None)
                arrays['mag_base_%s' % filt].append(None)
                arrays['mag_base_err_%s' % filt].append(None)

    return arrays


def cands_to_table(cands, fname):
    arrays = construct_arrays(cands)
    cols = []
    for colname, array in arrays.items():
        if colname == 'id':
            fmt = '25A'
        else:
            fmt = 'E'
        col = fits.Column(name=colname, format=fmt, array=array)
        cols.append(col)
    coldefs = fits.ColDefs(cols)
    hdu = fits.BinTableHDU.from_columns(coldefs)
    hdu.writeto('%s/%s' % (return_data_dir(), fname), overwrite=True)


def export_level4_ongoing_candidates():
    cands = db.session.query(CandidateLevel4).outerjoin(CandidateLevel3,
                                                        CandidateLevel4.id == CandidateLevel3.id). \
        filter(CandidateLevel3.t0_best + CandidateLevel3.tE_best > MJD_finish).\
        all()
    fname = 'level4_ongoing_candidates.fits'
    print('Exporting %i level4_ongoing candidates' % len(cands))
    cands_to_table(cands, fname)
    print('%s exported' % fname)


def export_level6_events():
    cands = CandidateLevel4.query.filter(CandidateLevel4.level5 == 't',
                                         CandidateLevel4.category == 'clear_microlensing').\
        all()
    fname = 'level6_events.fits'
    print('Exporting %i level6 events' % len(cands))
    cands_to_table(cands, fname)
    print('%s exported' % fname)


if __name__ == '__main__':
    export_level4_ongoing_candidates()
    export_level6_events()
