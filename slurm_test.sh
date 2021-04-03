#!/bin/bash

PATH=/global/common/shared/das/container_proxy/:$PATH
a=$(squeue -u mmedford)
echo $a
squeue -u mmedford > slurm_test.txt