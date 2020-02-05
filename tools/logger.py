#!/usr/bin/python3
from datetime import datetime
from os import path
import traceback

try:
    from os import mkdirs  # for python3.5
except:
    from os import makedirs as mkdirs # for python3.6 and above

class Logger:
        # script_name: indentifying name to prefix the log file
        # config: the configuration object being used for the current run of
	#         whatever it is that we're logging
	def __init__(self, script_name, config):
		self.name = script_name
		self.last_year = datetime.now().year
		self.target_file = ""

		site = config["site_id"]
		self.log_root = config["paths"][site]["logging_path"]

		if not path.isdir( self.log_root ):
			mkdirs( self.log_root )

		self.log( "Initializing with configuration(s): {}".format(
		    config["metadata"]["name"]
		) )

	def log( self, msg ):
		t = datetime.now()
		year = t.year
		if year != self.last_year:
			self.last_year = year
			self.target_file = ""
		if not self.target_file:
			self.target_file = "{}{}_{}.log".format(
			    self.log_root, self.name, year
			)
		
		msg = "{}: {}\n".format(t, msg.rstrip())
		with open(self.target_file, "a") as f:
			f.write(msg)
	def log_exception( self, msg="" ):
		msg = "Caught Exception: {}\n".format( msg ) + traceback.format_exc()
		self.log( msg )
