#! /usr/bin/env python
"""
recalculate_chi2_stats_on_ulens.py
"""

from generate_ulens_sample import *


def _recalculate_chi2_stats(sibsFlag):

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        rank = 0
        size = 1

    data_dir = return_data_dir()
    my_stats_complete_fname = f'{data_dir}/ulens_samples/stats.{rank:02d}.txt'

    fname = return_ulens_data_fname('ulens_sample')
    if sibsFlag:
        fname = fname.replace('sample', 'sample.sibs')
    data = load_stacked_array(fname)
    lightcurve_data = []
    for i, d in enumerate(data):
        lightcurve_data.append(d)

    fname = return_ulens_data_fname('ulens_sample_stats')
    if sibsFlag:
        fname = fname.replace('stats', 'stats.sibs')
    stats = np.load(fname)

    idx_arr = np.arange(len(lightcurve_data))
    my_idx_arr = np.array_split(idx_arr, size)[rank]
    my_data = np.array_split(data, size)[rank]

    my_chi_squared_inside_level3_arr = []
    my_chi_squared_outside_level3_arr = []
    my_num_days_inside_level3_arr = []
    my_num_days_outside_level3_arr = []
    my_delta_hmjd_outside_level3_arr = []
    for i, d in enumerate(my_data):
        hmjd = d[:, 0]
        mag = d[:, 1]
        magerr = d[:, 2]
        idx = my_idx_arr[i]
        t0 = stats['t0_level3'][idx]
        tE = stats['tE_level3'][idx]

        data = calculate_chi_squared_inside_outside(hmjd=hmjd,
                                                    mag=mag,
                                                    magerr=magerr,
                                                    t0=t0,
                                                    tE=tE,
                                                    tE_factor=2)
        chi_squared_inside, chi_squared_outside, num_days_inside, num_days_outside, delta_hmjd_outside = data
        my_chi_squared_inside_level3_arr.append(chi_squared_inside)
        my_chi_squared_outside_level3_arr.append(chi_squared_outside)
        my_num_days_inside_level3_arr.append(num_days_inside)
        my_num_days_outside_level3_arr.append(num_days_outside)
        my_delta_hmjd_outside_level3_arr.append(delta_hmjd_outside)

        with open(my_stats_complete_fname, 'a+') as f:
            f.write(f'{i}\n')

    total_chi_squared_inside_level3_arr = comm.gather(my_chi_squared_inside_level3_arr, root=0)
    total_chi_squared_outside_level3_arr = comm.gather(my_chi_squared_outside_level3_arr, root=0)
    total_num_days_inside_level3_arr = comm.gather(my_num_days_inside_level3_arr, root=0)
    total_num_days_outside_level3_arr = comm.gather(my_num_days_outside_level3_arr, root=0)
    total_delta_hmjd_outside_level3_arr = comm.gather(my_delta_hmjd_outside_level3_arr, root=0)

    if rank == 0:
        chi_squared_inside_level3_arr = list(itertools.chain(*total_chi_squared_inside_level3_arr))
        chi_squared_outside_level3_arr = list(itertools.chain(*total_chi_squared_outside_level3_arr))
        num_days_inside_level3_arr = list(itertools.chain(*total_num_days_inside_level3_arr))
        num_days_outside_level3_arr = list(itertools.chain(*total_num_days_outside_level3_arr))
        delta_hmjd_outside_level3_arr = list(itertools.chain(*total_delta_hmjd_outside_level3_arr))

        fname = 'updated_chi2_flat.npz'
        if sibsFlag:
            fname = fname.replace('.npz', '.sibs.npz')
        np.savez(fname,
                 chi_squared_inside_level3=chi_squared_inside_level3_arr,
                 chi_squared_outside_level3=chi_squared_outside_level3_arr,
                 num_days_inside_level3=num_days_inside_level3_arr,
                 num_days_outside_level3=num_days_outside_level3_arr,
                 delta_hmjd_outside_level3=delta_hmjd_outside_level3_arr)


def recalculate_chi2_stats():
    # run consolidate lightcurves first

    if 'SLURMD_NODENAME' in os.environ:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
    else:
        rank = 0

    data_dir = return_data_dir()
    my_stats_complete_fname = f'{data_dir}/ulens_samples/stats.{rank:02d}.txt'
    if os.path.exists(my_stats_complete_fname):
        os.remove(my_stats_complete_fname)

    _recalculate_chi2_stats(sibsFlag=False)
    _recalculate_chi2_stats(sibsFlag=True)


if __name__ == '__main__':
    recalculate_chi2_stats()
