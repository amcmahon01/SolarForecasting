# create daily .csv files
# Use standard config_handler with .yaml config to parse arguments
from config_handler import handle_config
from datetime import datetime,timezone
import pandas as pd
import pysolar.solar as ps
import sys
import time
# note: pathlibs Path object have their own open method, use that for python <3.6
from pathlib import Path

if __name__ == "__main__":
    cp = handle_config( 
      metadata={"invoking_script":"GHIdaily"}, header="diagnostics"
    )
    site = cp["site_id"]
    config = cp['downloader']

    paths = cp['paths']
    logpath = Path(paths['logging_path'])
    logpath.mkdir(exist_ok=True)
    GHIpath = Path(paths['raw_GHI_path'])
    start_date=int(cp['diagnostics']['start_date'])
    end_date=int(cp['diagnostics']['end_date'])
    force_overwrite=int(cp['diagnostics']['force_overwrite'])
    
    GHIsensors = {}
    for GHIsensor in sorted(cp['GHI_sensors'].keys()):
        GHIsensor = GHIsensor.upper()
        GHIsensors[GHIsensor]=cp['GHI_sensors'][GHIsensor]

        dest = Path(GHIpath,GHIsensor)
        if not dest.is_dir():
            continue
        print(dest.name)
        datedirs=dest.glob("20[0-9][0-9][0-1][0-9][0-3][0-9]")
        print("Available dates:")
        for datedir in sorted(datedirs):
            if ( not datedir.is_dir() ):
                continue
            if ( int(datedir.name) < start_date ) or ( int(datedir.name) > end_date ):
                print("skipping %s " % datedir.name)
                continue
            print("processing %s " % datedir.name)
            outfile=Path(dest,datedir.name+".csv")
            if ( outfile.exists() and not force_overwrite ):
                print("outfile already exists and no force_overwrite: %s" % outfile.name)
                continue
            ghifiles=datedir.glob("*.csv")
            count=0
            for ghifile in sorted(ghifiles):
                dfghi=pd.read_csv(ghifile,index_col="Record",usecols=["Record","TimeStamp","SlrW"],parse_dates=["TimeStamp"])
                count+=1
                if ( count == 1 ):
                    dfghiday=dfghi
                else:
                    dfghiday=dfghiday.append(dfghi)
            dfghiday=dfghiday.drop_duplicates()
            dfghiday.to_csv(outfile)
