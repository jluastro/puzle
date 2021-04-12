#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi

for fname in dec*classifications.csv;
    do uploaded=${fname%.csv}.uploaded
    if [ -f "$uploaded" ]; then
      echo "Skipping ${fname}"
    else
      echo "Uploading ${fname}";
      check=$($db_con -c "\copy ps1_psc (obj_id,ra_stack,dec_stack,rf_score,quality_flag) from ${fname} delimiter ',' csv header");
      if [[ 0 -eq $? ]]; then
        echo "-- Success";
        touch $uploaded
      else
        echo "-- Fail";
      fi
    fi
done
