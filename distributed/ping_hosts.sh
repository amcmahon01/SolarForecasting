#!/bin/bash
cut -d ',' -f 1 dist_hosts.csv | while read line; do
  ping $line -c 3
done
