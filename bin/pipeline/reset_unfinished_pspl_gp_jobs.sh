#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
  slurm_cond=""
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
  job_ids=$(squeue -u mmedford --noheader --format="%i")
  job_ids=$(echo $job_ids | tr " " ,)

  if [ -z "$job_ids" ]
  then
    slurm_cond="and slurm_job_id is NULL"
  else
    slurm_cond="and (slurm_job_id not in ($job_ids) OR (slurm_job_id is NULL))"
  fi
fi
$db_con -c "UPDATE candidate_level4 SET pspl_gp_fit_started='f', pspl_gp_fit_finished='f', pspl_gp_fit_datetime_started=NULL, pspl_gp_fit_datetime_finished=NULL, slurm_job_id=NULL, node=NULL WHERE pspl_gp_fit_started='t' and pspl_gp_fit_finished='f' $slurm_cond";
