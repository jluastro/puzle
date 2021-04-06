#! /usr/bin/env python
"""
convert_dr5_to_ascii.py
"""

import glob
import os
import numpy as np
import pyarrow.parquet as pq


def sort_files_into_folders(path='/global/cfs/cdirs/uLens/ZTF/DR5'):
    files = glob.glob(f'{path}/*parquet')
    files.sort()
    for i, file in enumerate(files):
        if i % 10 == 0:
            print(i, len(files))
        file_base = os.path.basename(file)
        field = 'field' + file_base.split('_')[1]
        field_dir = f'{path}/{field}'
        if not os.path.exists(field_dir):
            os.makedirs(field_dir)
        file_new = f'{field_dir}/{file_base}'
        os.rename(file, file_new)


def parquet_to_asciilc(inpath, outdir='/global/cfs/cdirs/uLens/ZTF/DR5'):
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
    ds = pq.ParquetDataset(inpath, use_legacy_dataset=False)
    df = ds.read().to_pandas()
    df.sort_values(['filterid', 'rcid', 'objdec'],
                   ascending=[False, True, True],
                   inplace=True)
    fields = df.fieldid.unique()
    assert len(fields) == 1
    field = fields[0]
    ramin = df.objra.min()
    ramax = df.objra.max()
    decmin = df.objdec.min()
    decmax = df.objdec.max()
    outname = os.path.join(outdir,
                           'field{:06d}_ra{:.5f}to{:.5f}_dec{:.5f}to{:.5f}.txt'.format(
                               field, ramin, ramax, decmin, decmax))
    counter = 0
    with open(outname, 'w') as fh:
        for row in df.itertuples():
            fh.write("# %17d %3d %1d %4d %2d %9.5f %9.5f\n" %
                     (row.objectid, row.nepochs, row.filterid, row.fieldid,
                      row.rcid, row.objra, row.objdec))
            idx = np.argsort(row.hmjd)
            for i in idx:
                fh.write("  %13.5f %6.3f %5.3f %6.3f %5d\n" %
                         (row.hmjd[i], row.mag[i], row.magerr[i],
                          row.clrcoeff[i], row.catflags[i]))
            counter += 1
            if counter == 100:
                break
    return


def convert_DR5_to_ascii(path='/global/cfs/cdirs/uLens/ZTF/DR5'):
    folders = glob.glob(f'{path}/field*')
    folders.sort()

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    my_folders = np.array_split(folders, size)[rank]

    for i, folder in enumerate(my_folders):
        print('%i) Converting %s (%i / %i)' % (rank, folder, i, len(my_folders)))
        parquet_to_asciilc(inpath=folder,
                           outdir=path)


if __name__ == '__main__':
    convert_DR5_to_ascii()
