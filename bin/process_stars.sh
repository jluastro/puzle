#!/bin/bash
#SBATCH --account=m2218
#SBATCH --image=registry.services.nersc.gov/mmedford/puzle:latest
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=2
#SBATCH --time=00:30:00
#SBATCH --job-name=stars
#SBATCH --output=stars.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

export PROXY_SOCKET=/tmp/${USER}.${SLURM_JOB_ID}.sock
/global/common/shared/das/container_proxy/server.py &
CPID=$!

srun -N 2 -n 64 shifter --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4;/global/cfs/cdirs/uLens/PS1_PSC:/home/puzle/data/PS1_PSC;/global/u2/m/mmedford/puzle/data/ulensdb:/home/puzle/data/ulensdb" python /home/puzle/process_stars.py

kill $CPID

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
