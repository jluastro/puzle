#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=debug
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

srun -N 1 -n 5 shifter --image=registry.services.nersc.gov/mmedford/puzle:latest --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4" --volume="/global/cfs/cdirs/uLens/PS1_PSC:/home/puzle/data/PS1_PSC" --volume="/global/u2/m/mmedford/puzle/data/ulensdb:/home/puzle/data/ulensdb" /home/puzle/mpi_test.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
