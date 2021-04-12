#!/bin/bash
#SBATCH --account=ulens
#SBATCH --qos=premium
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=convert
#SBATCH --output=convert.%j.out
echo "---------------------------"
echo "Job id = $SLURM_JOBID"
echo "Proc id = $SLURM_PROCID"
hostname
date
echo "---------------------------"

conda activate pyarrow
python /global/homes/m/mmedford/puzle/bin/pipeline/convert_dr5_to_ascii.py

echo "---------------------------"
date
echo "All done!"
echo "---------------------------"
