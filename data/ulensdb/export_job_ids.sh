#! /bin/bash
fname=/global/homes/m/mmedford/puzle/data/ulensdb/job_ids.txt
date > $fname
squeue --noheader -u mmedford --format="%i" >> $fname
