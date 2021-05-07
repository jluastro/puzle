#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=pspl_gp
#SBATCH --output=pspl_gp.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

module load cmake/3.18.2
module unload craype-hugepages2M
export LD_LIBRARY_PATH=/global/cfs/cdirs/uLens/code/src/MultiNest/lib:$LD_LIBRARY_PATH

conda activate puzle
srun -N 1 -n 32 python /global/homes/m/mmedford/puzle/bin/pipeline/fit_level4_candidates_to_pspl_gp.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
