Fetch images from the cameras at prescribed intervals.

	::
	Usage: python3 image_downloader.py [-c configuration_header]
		in particular, '-c alb_dl' and '-c bnl_dl' are of use
The definition of cameras and intervals are specified in the configuration file.

Dependencies: pysolar
Other dependencies come with anaconda3.

Installation
------------

	::
	cp image_downloader.sh.dist ~nowcast/run/image_downloader.sh 

And edit, if needed.

	::
	chmod +x ~nowcast/run/image_downloader.sh
