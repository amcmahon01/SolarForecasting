# plot daily box-whisker graphs of GHI data
# Use standard config_handler with .yaml config to parse arguments
from config_handler import handle_config
from datetime import datetime,timezone,date
import pandas as pd
import matplotlib.pyplot as plt
import pysolar.solar as ps
import sys
import time
# note: pathlibs Path object have their own open method, use that for python <3.6
from pathlib import Path

if __name__ == "__main__":
    cp = handle_config( 
      metadata={"invoking_script":"GHIplot"}, header="diagnostics"
    )
    site = cp["site_id"]
    config = cp['downloader']

    paths = cp['paths']
    logpath = Path(paths['logging_path'])
    logpath.mkdir(exist_ok=True)
    GHIpath = Path(paths['raw_GHI_path'])
    start_date=int(cp['diagnostics']['start_date'])
    end_date=int(cp['diagnostics']['end_date'])

    GHIsensors = {}
    for GHIsensor in sorted(cp['GHI_sensors'].keys()):
        GHIsensor = GHIsensor.upper()
        GHIsensors[GHIsensor]=cp['GHI_sensors'][GHIsensor]

        source = Path(GHIpath,GHIsensor)
        if not source.is_dir():
            continue
        print(source.name)
        datefiles=source.glob("20[0-9][0-9][0-1][0-9][0-3][0-9].csv")
        print("Available dates:")
        for datefile in sorted(datefiles):
            if ( not datefile.is_file() ):
                continue
            datestr=datefile.stem
            if ( int(datestr) < start_date ) or ( int(datestr) > end_date ):
                continue
            outfile=Path(source,datestr+".png")
            df=pd.read_csv(datefile,index_col="Record",usecols=["Record","TimeStamp","SlrW"],parse_dates=["TimeStamp"])
            isodate=date(int(datestr[0:4]),int(datestr[4:6]),int(datestr[6:8])).isoformat()
            ax=df.plot(x='TimeStamp',y='SlrW',title="GHI sensor: %s, %sZ" % (GHIsensor,isodate))
# ,ylabel="GHI [W.m^-2]"            
            ax.figure.show()
