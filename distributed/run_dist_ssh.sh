#!/bin/bash

cut -d ',' -f 1 dist_hosts.csv | while read host; do
  echo $host
  cmd='cd /share/solardb-data/code/SolarForecasting/distributed; python hello_solar.py config_' + $host + '.conf  > ~/data/runlog.txt'
  ssh nowcast@$host "$cmd" > ~/logs/$host.txt &
done
