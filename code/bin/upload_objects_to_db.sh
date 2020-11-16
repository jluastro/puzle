#!/usr/bin/env bash

for f in field*objects; do
  echo "Uploading $f";
  f_db=${f}_db;
  f_comp=${f_db}.comp;

  if test -f $f_comp; then
    echo "--Skipping due to $f_comp";
    continue
  fi

	sed "s/$/,$f/" $f > ${f_db};
	psql -U ulens_admin -d ulens -h nerscdb03.nersc.gov -c "\copy object (id, nepochs, filterid, fieldid, rcid, ra, dec, lightcurve_position, lightcurve_filename) FROM '$PWD/$f_db' WITH DELIMITER ',' CSV HEADER;"
	touch $f_comp
	rm $f_db
done
