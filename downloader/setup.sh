#!/bin/sh
# better to do this in a python module that reads the dirs from the config file.
: ${SITE:="alb"}
sudo mkdir -p /${SITE}/data/{cache,latest,images}
sudo chown nowcast:nowcast /${SITE}/data/{cache,latest,images}
