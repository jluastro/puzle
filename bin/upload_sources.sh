#!/bin/bash

for job_id in $(psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "COPY (SELECT id FROM source_ingest_job WHERE finished='t' AND uploaded='f') TO STDOUT");
    do echo "Uploading sources ${job_id}";
    folder=sources_${job_id:0:3};
    fname=${folder}/sources.${job_id}.txt;
    psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "\copy source (id,object_id_g,object_id_r,object_id_i,lightcurve_position_g,lightcurve_position_r,lightcurve_position_i,lightcurve_filename,ra,dec,ingest_job_id) from ${fname} delimiter ',' csv header NULL as 'None'";
    psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "update source_ingest_job set uploaded='t' where id=${job_id}";
done
