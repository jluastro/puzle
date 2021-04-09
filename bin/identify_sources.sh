#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=13:00:00
#SBATCH --job-name=sources
#SBATCH --output=sources.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

conda activate puzle
cd /global/cfs/cdirs/uLens/ZTF/DR5
srun -N 1 -n 32 python /global/homes/m/mmedford/puzle/bin/identify_sources.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
