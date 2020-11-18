#!/bin/bash
#SBATCH --account=ulens
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=source_table
#SBATCH --output=source_table_%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

conda activate puzle
cd /global/cfs/cdirs/uLens/ZTF/DR3
srun -N 1 -n 32 python /global/homes/m/mmedford/puzle/bin/populate_source_table.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"