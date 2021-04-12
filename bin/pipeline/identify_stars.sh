#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=42:00:00
#SBATCH --job-name=stars
#SBATCH --output=stars.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

conda activate puzle
cd /global/cfs/cdirs/uLens/ZTF/DR5
srun -N 1 -n 32 python /global/homes/m/mmedford/puzle/bin/pipeline/identify_stars.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
