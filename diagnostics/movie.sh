#!/bin/bash
# this makes a 1 min long movie from all the .jpg images in a day.
# requires ffmpeg, which is in RPM Fusion repo and currently (2020-09-18) only
# installed on rpal.das.
# To make this useful pass site, camera and date on command-line,
# right now you can set these in the env.
: ${SITE:="bnl"}
: ${CAM:="HD1A"}
: ${DATE:="20200912"}
: ${OUTDIR:="${HOME}/tmp4"}
cd /net/solar-db.bnl.gov/home/nowcast/data/${SITE}/images/${CAM}
cat ${DATE}/${CAM}_*.jpg |ffmpeg -f image2pipe -c:v mjpeg -i - ${OUTDIR}/${CAM}_${DATE}.mpg
# use nowcast.das to ftp to ftp.nowcast
