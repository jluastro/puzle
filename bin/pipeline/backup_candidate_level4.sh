#!/bin/bash

if [ -z "$NERSC_HOST" ]
then
  db_con="psql ulens -U ulens_admin"
else
  db_con="psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov"
fi
echo "Creating copy of candidate_level4 table";
$db_con -c "CREATE TABLE candidate_level4_backup AS TABLE candidate_level4";
