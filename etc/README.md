#Automatic downloader systemd service configuration#
These systemd service config files `{image,GHI}_downloader_{alb,bnl}.service` should be
installed into the directory returned by:

`pkg-config systemd --variable=systemdsystemunitdir`
	
It runs the commands in
`/home/nowcast/run/{image,GHI}_downloader_{alb,bnl}.sh`.

**Important note:** These `bash` scripts must explicitly source
  `~/.bash_profile`, because the services run in a restricted
  environment. The profile must include setting the environment
  variable `PYTHONPATH="$HOME/release/tools"` so the `.py` scripts
  find the `config_handler` module. The profile also sources
  `~/.bashrc` which sets and activates the `py35` environment.

To enable at boot

`systemctl enable {image,GHI}_downloader_{alb,bnl}`
	
To start

`systemctl start {image,GHI}_downloader_{alb,bnl}`
	
The status is shown with

`systemctl status -l {image,GHI}_downloader_{alb,bnl}`
	
Note: This also shows the subprocesses started, and the last error
messages.

The process is supposed to restart with a 1 s delay when it dies and
disable itself if there are more than 5 restarts in a 10 s interval.
Killing a sub-process restarts automatically within 1 s.

#Daily cron jobs#

##GHIdaily.sh##
This runs [currently only on `nowcast`] `GHIdaily.py` to merge the
individual 31s snippets of GHI data, eliminates duplicates and stores
the result in a daily `.csv` file, that is then upload to
`ftp.nowcast.bnl.gov:/outgoing`.

The **Important note** above also applies here.
