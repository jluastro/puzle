#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi
for fname in dec*classifications.csv;
    do echo "Uploading ${fname}";
    $db_con -c "\copy ps1_psc (obj_id,ra_stack,dec_stack,rf_score,quality_flag) from ${fname} delimiter ',' csv header";
done
