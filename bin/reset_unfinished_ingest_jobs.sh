#!/bin/bash

psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "UPDATE source_ingest_job SET started='f', finished='f', datetime_started=NULL, datetime_finished=NULL, slurm_job_id=NULL, slurm_job_rank=NULL WHERE started='t' and finished='f'";
psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "UPDATE star_ingest_job SET started='f', finished='f', datetime_started=NULL, datetime_finished=NULL, slurm_job_id=NULL, slurm_job_rank=NULL WHERE started='t' and finished='f'";
