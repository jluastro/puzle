#!/bin/bash

a=$(squeue -u mmedford)
echo $a
squeue -u mmedford > slurm_test.txt