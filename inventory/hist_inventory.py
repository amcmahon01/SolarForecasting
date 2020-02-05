### inventory.py
# every hour count up the new photos taken by each camera 
#    and add an entry to inventory.csv
# TODO: redo inventory for given time range 
# 	(e.g if we sync data from another machine, 
# 	    update inventory for only those hours)
###

from datetime import datetime, timedelta
import os
from pathlib import Path
import pandas as pd
import time
import traceback

try:
    from config_handler import handle_config
    from logger import Logger
except:
    print( "Failed to import config_handler or logger.\n" + 
           "They are located in ../tools/\n" )
    print( traceback.format_exc() )
    exit( 1 )

cp = handle_config(
    metadata={"invoking_script":"hist_inventory"}, 
    header="inventory"
)
logger = Logger( "hist_inventory", cp )

### load configuration
SITE = cp["site_id"]
CAMERAS = cp["cameras"][SITE]["cameras"]
NUM_CAMERAS = len(CAMERAS)

# how long to sleep once we run out of images to log
SLEEP_LEN = cp["inventory"]["hist"]["sleep_time"]

ROOT = cp["paths"][SITE]["img_path"] # image directory
CACHE_ROOT = cp["paths"][SITE]["cache_path"] # cache directory

FOLDER_NAMES = [ROOT + x + "/" for x in CAMERAS]
CACHE_FOLDER_NAMES = [CACHE_ROOT + x + "/" for x in CAMERAS]

# location of the inventory.csv files
INVENTORY_ROOT = cp["paths"][SITE]["inventory_path"] 

# the code as is depends on the timestamp column being 'hour'
HEADER = "hour," + ",".join(CAMERAS).lower() # inventory.csv header
HEADER_COLS = ["hour"] + CAMERAS

default_s_time = "0000010100"
if cp["inventory"]["start_date"]:
	next_t = cp["inventory"]["start_date"]
	next_t = str(next_t) + default_s_time[len(str(next_t)):]
	next_t = datetime.strptime( next_t, "%Y%m%d%H" )
else:
	# default to the start of the current year
	next_t = datetime( datetime.now().year, 1, 1 )

if cp["inventory"]["end_date"]:
	end_t = cp["inventory"]["end_date"]
	end_t = str(end_t) + default_s_time[len(str(end_t)):]
	end_t = datetime.strptime( end_t, "%Y%m%d%H" )
else:
	end_t = None

### helper methods
# was the file created during target_hour?
def in_hour( file_name, target_hour ):
	return file_name[5:-8] == target_hour # fragile...

def get_inventory_name( year ):
	return INVENTORY_ROOT + "inventory_{}.csv".format(year)
	
def create_inventory_file_if_missing( year ):
	name = get_inventory_name( year )
	if not os.path.exists( name ):
		with open( name, "w" ) as inventory:
			 inventory.write( HEADER + "\n" )
	return name

def list_files( day ):
	file_lists = []
	for path in FOLDER_NAMES:
		p = path + day + "/"
		# if there are no records for a day
		# use a blank array
		if os.path.isdir( p ):
			file_lists.append( os.listdir(p) )
		else:
			file_lists.append( [] )
	return file_lists

def list_cached_files( ):
	file_lists = []
	for path in CACHE_FOLDER_NAMES:
		file_lists.append( os.listdir(path) )
	return file_lists

def full_path( folder, file_n, use_cache=False ):
	global day
	if use_cache:
		return os.path.join( folder + "/", file_n )
	else:
		return os.path.join( folder + day + "/", file_n )

def finish_inv():
	print( "Finished logging up to {}. Exiting.".format( end_t ) )
	exit(0)

# construct and return an array representing an entry in inventory*.csv
def count_folders( time_str, file_lists, use_cache=False ):
	folder_names = CACHE_FOLDER_NAMES if use_cache else FOLDER_NAMES
	count = [""] + ['0']*NUM_CAMERAS

	for i, folder in enumerate(folder_names):
		# filter file list for images created during the target hour
		file_list = [
		    f for f in file_lists[i]
		    if f[-18:-8] == time_str 
		    and os.path.isfile( full_path( folder, f, use_cache ) ) 
		]

		count[i+1] = str(len(file_list))
		if len(file_list) and not count[0]:
			count[0] = file_list[0][-18:-8] # fragile
	return count


# create the inventory directory if it doesn't exist
Path( INVENTORY_ROOT ).mkdir( parents=True, exist_ok=True )

# if there is an inventory file for the target year 
# start logging from the point where we left off earlier
inventory_name = create_inventory_file_if_missing( next_t.year )
old_data = pd.read_csv(inventory_name)
if len(old_data.index):
	t = old_data.loc[len(old_data.index)-1]['hour']
	next_t = datetime.strptime( str(t), "%Y%m%d%H" ) + timedelta(hours=1)

### main
last_day = None
last_year = next_t.year

file_lists = []
while True:
	try:
		counts = pd.DataFrame( columns=HEADER_COLS )

		max_t = datetime.now() # when are we sure there's no more data?
		last_success = next_t # the last hour where we found data
		last_counts = 0 # counts during last_success

		use_cache = False
		while True:
			if end_t and next_t >= end_t:
				counts = counts[0:last_counts]
				break

			if next_t >= max_t:
				next_t = last_success
				counts = counts[0:last_counts]
				
				if not use_cache:
					use_cache = True
					continue
				break
			
			if next_t.year != last_year:
				last_year = next_t.year
				break

			day = next_t.strftime( "%Y%m%d" )
			if day != last_day:
				if not use_cache:
					file_lists = list_files( day )
				else:
					file_lists = list_cached_files()

				last_day = next_t.strftime( "%Y%m%d" )

			next_update_str = next_t.strftime( "%Y%m%d%H" )

			# count the number of images taken by each camera
			# during the hour after next_t (UTC)
			count = count_folders(
			    next_update_str, file_lists, use_cache=use_cache
			)
			counts.loc[len(counts.index)] = count

			if count[0]:
				last_success = next_t
				last_counts = counts.index[-1]
			else:
				counts.loc[counts.index[-1]][0] = next_update_str
				logger.log( 
				    "Unable to find data for time " + 
				    next_t.strftime( "%Y-%m-%d %H:%M" )
				)

			next_t += timedelta( hours=1 )

		# write new counts and the name of the hour to inventory.csv
		counts.to_csv(inventory_name,mode='a',header=False,index=False)
	except Exception as e:
		logger.log_exception()
		print( "Caught exception {}".format(e) )
	finally:
		if end_t and next_t >= end_t:
			finish_inv()
		else:
			time.sleep(SLEEP_LEN)
