#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=export-sources
#SBATCH --output=export-sources.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

conda activate puzle
cd /global/cfs/cdirs/uLens/ZTF/DR5
srun -N 1 -n 32 python export_sources.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
