#! /usr/bin/env python
"""
convert_dr5_to_ascii.py
"""

import glob
import os
import numpy as np
import pandas as pd


def parquet_to_ascii(inpath, outdir='/global/cfs/cdirs/uLens/ZTF/DR5'):
    """
    Convert ZTF lightcurve Parquet dataset for one field to the pre-DR5 ascii format

    Parameters:
    -----------
    inpath: string
        path to field-level directory of Parquet files
        example: './data/ZTF/lc_dr5/0/field0350/'

    outdir: string
        path to directory for output of ascii text file for zort

    """
    files = glob.glob(f'{inpath}/*parquet')
    field = int(inpath.split('/')[-1].replace('field', ''))
    fname_tmp = f'{outdir}/field{field:06d}_tmp.txt'
    ra_arr = []
    dec_arr = []
    with open(fname_tmp, 'w') as fh:
        for f in files:
            df = pd.read_parquet(f, engine='pyarrow')
            for row in df.itertuples():
                fh.write("# %17d %3d %1d %4d %2d %9.5f %9.5f\n" %
                         (row.objectid, row.nepochs, row.filterid, row.fieldid,
                          row.rcid, row.objra, row.objdec))
                ra_arr.append(row.objra)
                dec_arr.append(row.objdec)
                idx = np.argsort(row.hmjd)
                for i in idx:
                    fh.write("  %13.5f %6.3f %5.3f %6.3f %5d\n" %
                             (row.hmjd[i], row.mag[i], row.magerr[i],
                              row.clrcoeff[i], row.catflags[i]))

    ramin = np.min(ra_arr)
    ramax = np.max(ra_arr)
    decmin = np.min(dec_arr)
    decmax = np.max(dec_arr)
    fname = os.path.join(outdir,
                         'field{:06d}_ra{:.5f}to{:.5f}_dec{:.5f}to{:.5f}.txt'.format(
                             field, ramin, ramax, decmin, decmax))
    os.rename(fname_tmp, fname)


def convert_DR5_to_ascii(path='/global/cfs/cdirs/uLens/ZTF/DR5'):
    folders = [f for f in glob.glob(f'{path}/field*') if '.' not in f]
    folders.sort()
    for i, folder in enumerate(folders):
        output_files = [f for f in glob.glob(f'{folder}*.*') if f != folder]
        if len(output_files) == 0:
            print('Converting %s (%i / %i)' % (folder, i, len(folders)))
            parquet_to_ascii(inpath=folder,
                             outdir=path)
        else:
            print('Skipping %s (%i / %i)' % (folder, i, len(folders)))


if __name__ == '__main__':
    convert_DR5_to_ascii()
