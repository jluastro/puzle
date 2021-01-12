#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi
$db_con -c "UPDATE source_ingest_job SET started='f', finished='f', datetime_started=NULL, datetime_finished=NULL, slurm_job_id=NULL, slurm_job_rank=NULL WHERE started='t' and finished='f'";
$db_con -c "UPDATE star_ingest_job SET started='f', finished='f', datetime_started=NULL, datetime_finished=NULL, slurm_job_id=NULL, slurm_job_rank=NULL WHERE started='t' and finished='f'";
