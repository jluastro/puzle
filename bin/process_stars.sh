#!/bin/bash
#SBATCH --account=ulens
#SBATCH --image=registry.services.nersc.gov/mmedford/puzle:v0.0.14
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=10
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=test.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"


echo $SLURM_JOBID >> $fname
srun -N 10 -n 320 shifter --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4;/global/cfs/cdirs/uLens/PS1_PSC:/home/puzle/data/PS1_PSC;/global/u2/m/mmedford/puzle/data/ulensdb:/home/puzle/data/ulensdb" python /home/puzle/test_ulensdb.py
sed -i "/${SLURM_JOBID}/d" "$fname"

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
