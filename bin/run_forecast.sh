#!/bin/bash
# this should really be changed to a python script that reads 
# the yaml config and passes it to the relevant processes.
# predict.py currently assumes model files are located in its CWD, 
# i.e. the scripted_processing directory, so need to CD there.
ME=$0
MYDIR=`dirname $ME`
MYDIR=`( cd $MYDIR; pwd )`
# try letting python find the rolling.so module in the tools
#: ${PYTHONPATH:="${HOME}/release/tools"}
# for now use symlink instead because there are differences between the
# stat_tools in scripted_processing (Andrew) and the one in "tools" (Theo)
: ${CONF:="$MYDIR/config.conf"}
: ${SITE:="bnl"}
: ${RELEASE:="${HOME}/release/code/scripted_processing"}
: ${PROCESSES:="preprocess generate_stitch extract_features GHI_preprocessing predict"}
: ${DATAROOT:="${HOME}/data/${SITE}"}
: ${SERVERROOT:="solar-db.bnl.gov:data/${SITE}"}
function die() {
    echo "$@" 1>2
    echo "usage: $ME StartDate NumDays" 1>2
    echo "       where StartDate in YYYYMMDD format" 1>2
    echo "       change env DATAROOT=$DATAROOT if data is to be written from a " 1>2
    echo "       different dir. tree" 1>2
    echo "       change env SERVERROOT=$SERVERROOT the raw data resides somewhere else" 1>2
    exit 1
}
# kludge DAYS line for old config.conf for now.
# expect two arguments for now: DAY1 NDAYS
case $# in
    2) DAY1=$1; NDAYS=$2;;
    *) die "need 2 arguments";;
esac
# if HOME is not /home/nowcast, need to update in config.conf
if [ ${HOME} != "/home/nowcast" ]
then
    ed $MYDIR/config.conf <<EOF
/^HOME=/c
HOME=${HOME}
.
,w
q
EOF
fi
#
# We are going one day at a time so this becomes the outer loop
T=`date -d $DAY1 +'%s'`
DATE=`date -d "@$T" +%Y%m%d`
DAYSTR="days=['$DATE']"
IDAY=1
while [ $IDAY -le $NDAYS ]
do
    # now replace 'days=' line in $MYDIR/config.conf
    ed $MYDIR/config.conf <<EOF
/^days=/c
$DAYSTR
.
,w
q
EOF
    
    # put the rest in a python script reading from config.conf,
    # better yet config.yaml
    
    # copy input raw image data
    #
    ( cd $DATAROOT/images
      # need to grep all_cams from config.conf and mkdir if necessary
      eval `grep "^all_cams=" $CONF|sed 's/\[/\(/'|sed 's/\]/\)/'|sed 's/,/ /g'|sed 's/all_cams=/CAMS=/'`
      for CAM in "${CAMS[@]}"; do ( mkdir -p $CAM; cd $CAM; rsync -au ${SERVERROOT}/images/$CAM/$DATE . ; ) ; done
    )
    GHIDIR=`date -d $DATE +%Y-%m`
    GHI_new="${HOME}/data/${SITE}/GHI_new/${GHIDIR}/GHI_25.npz"
    
    # Need to run GHI_processing if .npz file doesn't exist yet.

    if [ ! -r "$GHI_new" ]
    then
	( cd $DATAROOT/GHI
	  rsync -au ${SERVERROOT}/GHI/${GHIDIR} . )
    fi
    
    LOGDIR=${DATAROOT}/log/${DATE}
    mkdir -p $LOGDIR
    for P in ${PROCESSES}
    do
	cd $RELEASE
	time python3 ./${P}.py ${CONF} > ${LOGDIR}/${P}.log 2>&1
    done
    # now copy results back: stitch, feature, forecast, log
    # might want to use --remove-source-files on rsync to remove successfully copied
    # files
    ( cd $DATAROOT
      OUTDIRS=( stitch feature forecast )
      for ODIR in "${OUTDIRS[@]}"; do ( cd $ODIR; rsync -au $DATE/ $SERVERROOT/$ODIR/$DATE/; ) ; done
    )
    ( cd $DATAROOT/log; rsync -au ${DATE} $SERVERROOT/log/ )
    IDAY=`expr $IDAY + 1`
    T=`expr $T + 86400`
    DATE=`date -d "@$T" +%Y%m%d`
    DAYSTR="days=['$DATE']"
done
