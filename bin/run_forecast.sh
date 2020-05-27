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
: ${PROCESSES:="preprocess generate_stitch extract_features predict"}
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
T=`date -d $DAY1 +'%s'`
DAYSTR="days=['$DAY1'"
IDAY=1
while [ $IDAY -lt $NDAYS ]
do
    IDAY=`expr $IDAY + 1`
    T=`expr $T + 86400`
    DATE=`date -d "@$T" +%Y%m%d`
    DAYSTR=${DAYSTR}",'$DATE'"
done
DAYSTR=${DAYSTR}"]"
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
declare -a DAYS
eval `echo $DAYSTR |sed 's/\[/\(/'|sed 's/\]/\)/'|sed 's/,/ /g'|sed 's/days=/DAYS=/'`
( cd $DATAROOT/images
  # need to grep all_cams from config.conf and mkdir if necessary
  eval `grep "^all_cams=" $CONF|sed 's/\[/\(/'|sed 's/\]/\)/'|sed 's/,/ /g'|sed 's/all_cams=/CAMS=/'`
for CAM in "${CAMS[@]}"; do ( mkdir -p $CAM; cd $CAM; for DAY in "${DAYS[@]}"; do rsync -au ${SERVERROOT}/images/$CAM/$DAY .; done ; ) ; done
)
GHIDIRS=`{
    for DAY in "${DAYS[@]}"
    do
	echo $DAY|cut -c1-6
    done
    } | sort |uniq`
for GHIDIR in $GHIDIRS
do
    GHI_new="${HOME}/data/${SITE}/GHI_new/${GHIDIR}/GHI_25.npz"
    ( cd $DATAROOT/GHI
      rsync -au ${SERVERROOT}/GHI/${GHIDIR} . )
done

# Need to run GHI_processing if .npz file doesn't exist yet.

[ -r "$GHI_new" ] || PROCESSES="GHI_preprocessing $PROCESSES"

LOGDIR=${DATAROOT}/log/${DAY1}_${NDAYS}
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
  for ODIR in "${OUTDIRS[@]}"; do ( cd $ODIR; for DAY in "${DAYS[@]}"; do rsync -au $DAY $SERVERROOT/$ODIR/$DAY; done; ) ; done
)
( cd $DATAROOT/log; rsync -au ${DAY1}_${NDAYS} $SERVERROOT/log/ )