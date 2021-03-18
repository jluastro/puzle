#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi
for job_id in $($db_con -c "COPY (SELECT source_ingest_job_id FROM star_ingest_job WHERE started='t' AND finished='t' AND uploaded='f') TO STDOUT");
    do echo "Uploading stars ${job_id}";
    folder=stars_${job_id:0:3};
    fname=${folder}/stars.$(printf %06d $job_id).txt;
    $db_con -c "\copy star (id,ra,dec,ingest_job_id,source_ids) from ${fname} delimiter ',' csv header NULL as 'None'";
    $db_con -c "update star_ingest_job set uploaded='t' where source_ingest_job_id=${job_id}";
done
