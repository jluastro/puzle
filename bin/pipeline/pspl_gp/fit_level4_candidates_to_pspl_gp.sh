#!/bin/bash
#SBATCH --account=m2218
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=25
#SBATCH --time=48:00:00
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
nodelist=$(srun -N 1 -n 1 python /global/homes/m/mmedford/puzle/bin/pipeline/parse_nersc_nodelist.py)
echo "Nodes = $nodelist"

cd /global/homes/m/mmedford/puzle/bin/pipeline/pspl_gp
for node_name in $nodelist; do
  echo "Launching ${node_name}"
  for i in `seq 0 3`; do
    echo "Launching ${node_name} - ${i}"
    node_name_ext="$node_name.$i"
    fname_log="/global/homes/m/mmedford/puzle/bin/pipeline/pspl_gp/pspl_gp.$SLURM_JOBID.$node_name_ext.log"
    srun -N 1 -n 8 -w $node_name python /global/homes/m/mmedford/puzle/bin/pipeline/fit_level4_candidates_to_pspl_gp.py $SLURM_JOBID $node_name_ext > $fname_log &
    sleep 5
  done
done;
wait

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"