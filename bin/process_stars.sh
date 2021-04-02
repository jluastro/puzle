#!/bin/bash
#SBATCH --account=m2218
#SBATCH --image=registry.services.nersc.gov/mmedford/puzle:latest
#SBATCH --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4"
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=stars
#SBATCH --output=stars.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

srun -N 1 -n 32 shifter python /home/puzle/process_stars.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
