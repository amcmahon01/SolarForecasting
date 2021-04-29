#!/bin/bash
# this mirrors the albany GHI dirs to ftp.nowcast.bnl.gov
cd $HOME/data/alb/GHI
lftp ftp.nowcast.bnl.gov <<EOF
cd outgoing/alb/GHI
mirror -Rn -i 2020[01][0-9][0-3][0-9]\.csv ./ ./

EOF
