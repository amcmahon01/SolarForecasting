from datetime import datetime
import matplotlib as mpl
import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import traceback

try:
    from config_handler import handle_config
    from logger import Logger
except:
    print( "Failed to import config_handler or logger\n" + 
           "They are located in ../tools/\n" )
    print( traceback.format_exc() )
    exit(1)

cp = handle_config( metadata={"invoking_script":"plot_inventory"}, header="inventory" )
logger = Logger( "plot_inventory", cp )

SITE = cp["site_id"]
# slightly shift plotted y values so you can zoom in
# and see all the cameras at a given point
JITTER_STR = cp["inventory"]["plot"]["jitter"]
ALPHA = cp["inventory"]["plot"]["alpha"] # low alpha looks bad but has its uses
CAMERAS = cp["cameras"][SITE]["cameras"]
NUM_CAMERAS = len(CAMERAS)
ROOT = cp["paths"][SITE]["inventory_path"]
SAVE_LOC = ROOT + "inventory_plot.png" # unused
COLORS = cp["inventory"]["plot"]["plot_colors"]
SIZE = cp["inventory"]["plot"]["plot_size"]

if "start_date" in cp["inventory"] and cp["inventory"]["start_date"]:
    start_time = cp["inventory"]["start_date"]
else:
    start_time = datetime.now().year
if "end_date" in cp["inventory"] and cp["inventory"]["end_date"]:
    end_time = cp["inventory"]["end_date"]
else:
    end_time = datetime.now().strftime( "%Y%m%d%H" )

default_s_time = "0000010100"
# default_e_time = "0000123123"
start_time = str(start_time) + default_s_time[len(str(start_time)):]
end_time = str(end_time) + default_s_time[len(str(end_time)):]

start_date = datetime.strptime( start_time, "%Y%m%d%H" )
end_date = datetime.strptime( end_time, "%Y%m%d%H" )

# hide some cameras?
mask = [True]*NUM_CAMERAS
if "cameras_to_show" in cp["inventory"]["plot"] and cp["inventory"]["plot"]["cameras_to_show"]:
    mask = [i in cp["inventory"]["plot"]["cameras_to_show"] for i in range(1, NUM_CAMERAS+1)]

# slightly offset elements of arr because all the points
# in my scatter plot are directly on top of one another
def basic_jitter(arr):
        a = arr + np.random.randn(len(arr)) * JITTER_STR
        a[a<0] = 0
        a[a>120] = 120
        return a

years = range(start_date.year, end_date.year+1)
min_hour = int(start_date.strftime( "%Y%m%d%H" ))
max_hour = int(end_date.strftime( "%Y%m%d%H" ))

inventory_files = []
for y in years:
    inventory_name = "{}inventory_{}.csv".format(ROOT, y)
    if not os.path.exists( inventory_name ):
        print( "Warning: invalid inventory file " + inventory_name )
        logger.log( "Warning: attempted to read invalid inventory file " + inventory_name )
        continue
    inventory_files.append( inventory_name )
if not len(inventory_files):
    print( "Error: no valid inventory files found in [{}]".format( ", ".join([str(y) for y in years]) ) )
    logger.log( "Error: no valid inventory files found in [{}]".format( ", ".join([str(y) for y in years]) ) )
    exit(1)

counts = pd.DataFrame()
for name in inventory_files:
    counts = pd.concat( [counts, pd.read_csv(name)], ignore_index=True )

# are string timestamps have the same ordering as actual times
counts = counts[counts.hour > min_hour]
counts = counts[counts.hour < max_hour]
counts = counts.reset_index(drop=True)
first_hour = str(int(counts['hour'][0]))
last_hour = str(int(counts['hour'][counts.index[-1]]))

# x axis index is a list of datetimes
xs = pd.to_datetime(counts.loc[:,'hour'].astype(int),format="%Y%m%d%H").tolist()

cams = counts.columns.tolist()
cams.remove( 'hour' )
cams.sort()
data = [ counts[cam].tolist() for cam in cams ]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylim( [-5, 125] )

### This whole block is setting up properly displaying dates on the x-axis
### The automatic formatting for dates get bad if you zoom in, so I overrode it
### I sort of tried to make it less fragile, but it'll likely break if we update
plt.xticks(rotation=15)
x_locator = dates.AutoDateLocator()
ax.xaxis.set_major_locator( x_locator )
# print( dates.AutoDateFormatter( x_locator ).__dict__ )
formatter = dates.AutoDateFormatter( x_locator )
for k,v in formatter.scaled.items():
	if v == "%H:%M:%S":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
	elif v == "%H:%M:%S.%f":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
	elif v == "%d %H:%M":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
	elif v == "%M:%S.%f":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
	elif v == "%m-%d %H":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
	elif v == "%H:%M:%S.%f":
		formatter.scaled[k] = "%b %d %Y %H:%M:%S"
ax.xaxis.set_major_formatter( formatter ) 
### end

for i, cam in enumerate(data):
	if mask[i]:
		ax.scatter( xs, basic_jitter(cam), c=COLORS[i], edgecolors=COLORS[i],  s=float(SIZE), alpha=ALPHA )

plt.title( "Number of images created by each camera per hour" )
plt.xlabel( "Date" )
plt.ylabel( "Num images saved" )
plt.legend( [c.upper() for (c, b) in zip(cams, mask) if b], loc="upper right" )

plt.show()
