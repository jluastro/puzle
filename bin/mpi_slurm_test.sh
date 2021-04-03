#!/bin/bash
#SBATCH --account=m2218
#SBATCH --image=registry.services.nersc.gov/mmedford/puzle:latest
#SBATCH --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4"
#SBATCH --volume="/global/cfs/cdirs/uLens/PS1_PSC:/home/puzle/data/PS1_PSC"
#SBATCH --volume="/global/u2/m/mmedford/puzle/data/ulensdb:/home/puzle/data/ulensdb"
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

ls /global/common/shared/das/container_proxy

export PROXY_SOCKET=/tmp/${USER}.${SLURM_JOB_ID}.sock
/global/common/shared/das/container_proxy/server.py &
CPID=$!

srun -N 1 -n 1 shifter bash /home/puzle/slurm_test.sh
srun -N 1 -n 5 shifter python /home/puzle/mpi_slurm_test.py

kill $CPID

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
