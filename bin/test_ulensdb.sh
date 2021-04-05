#!/bin/bash
#SBATCH --account=m2218
#SBATCH --image=registry.services.nersc.gov/mmedford/puzle:v0.1.9
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=test
#SBATCH --output=test.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

export PROXY_SOCKET=/tmp/${USER}.${SLURM_JOB_ID}.sock
/global/common/shared/das/container_proxy/server.py &
CPID=$!

srun -N 1 -n 32 shifter --volume="/global/cfs/cdirs/uLens/ZTF/DR4:/home/puzle/data/DR4;/global/cfs/cdirs/uLens/PS1_PSC:/home/puzle/data/PS1_PSC;/global/u2/m/mmedford/puzle/data/ulensdb:/home/puzle/data/ulensdb" python /home/puzle/test_ulensdb.py

kill $CPID

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
