#!/bin/bash

db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
for job_id in $($db_con -c "COPY (SELECT source_ingest_job_id FROM star_ingest_job WHERE finished='t' AND uploaded='f') TO STDOUT");
    do echo "Uploading stars ${job_id}";
    folder=stars_${job_id:0:3};
    fname=${folder}/stars.${job_id}.txt;
    $db_con -c "\copy star (ra,dec,ingest_job_id,source_ids) from ${fname} delimiter ',' csv header NULL as 'None'";
    $db_con -c "update star_ingest_job set uploaded='t' where source_ingest_job_id=${job_id}";
done
