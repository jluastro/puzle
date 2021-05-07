#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=regular
#SBATCH --constraint=debug
#SBATCH --nodes=6
#SBATCH --time=00:30:00
#SBATCH --job-name=pspl_gp
#SBATCH --output=pspl_gp.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

module load cmake/3.18.2
module unload craype-hugepages2M
export LD_LIBRARY_PATH=/global/cfs/cdirs/uLens/code/src/MultiNest/lib:$LD_LIBRARY_PATH

conda activate puzle

nodelist=$(python ~/puzle/bin/pipeline/parse_nersc_nodelist.py)
for node_name in $nodelist; do
  srun -N 1 -n 32 python /global/homes/m/mmedford/puzle/bin/pipeline/fit_level4_candidates_to_pspl_gp.py &
done

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"