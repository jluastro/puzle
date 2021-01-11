#!/bin/bash

for job_id in $(psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "COPY (SELECT source_ingest_job_id FROM star_ingest_job WHERE finished='t' AND uploaded='f') TO STDOUT");
    do echo "Uploading stars ${job_id}";
    folder=stars_${job_id:0:3};
    fname=${folder}/stars.${job_id}.txt;
    psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "\copy star (ra,dec,ingest_job_id,source_ids) from ${fname} delimiter ',' csv header NULL as 'None'";
    psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "update star_ingest_job set uploaded='t' where source_ingest_job_id=${job_id}";
done
