#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi
for job_id in $($db_con -c "COPY (SELECT id FROM source_ingest_job WHERE started='t' AND finished='t' AND uploaded='f') TO STDOUT");
    do echo "Uploading sources ${job_id}";
    folder=sources_${job_id:0:3};
    fname=${folder}/sources.$(printf %06d $job_id).txt;
    $db_con -c "\copy source (id,object_id_g,object_id_r,object_id_i,lightcurve_position_g,lightcurve_position_r,lightcurve_position_i,lightcurve_filename,ra,dec,ingest_job_id) from ${fname} delimiter ',' csv header NULL as 'None'";
    $db_con -c "update source_ingest_job set uploaded='t' where id=${job_id}";
done
