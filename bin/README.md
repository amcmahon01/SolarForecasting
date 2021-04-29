# Obsoleted by `operational` branch #

Most of this is likely superceded by the reorganization in the
`operational` branch.

-------------------------------------------------------------------------------

These `.sh` files are really `bash` scripts and should be run through `bash`.

# setup.sh #
This installs all the prerequisites on a blank `nowcast` user home-dir. It should be run out of that directory as the user `nowcast`. Since the `git` release is also not installed yet. You have to download this script first:

```bash
cd # go home
wget https://raw.githubusercontent.com/BNL-NowCasting/SolarForecasting/master/bin/setup.sh
bash -x setup.sh 2>&1 |tee setup.log
```
	
It downloads and installs a `anaconda3` if it doesn't exist and creates a `py35` virtual environment which is then made the default in `~nowcast/.bashrc`

It then downloads a specific release version of the SolarForecasting distribution (e.g. `1.1`) and creates a symlink `release` that points to that version.

Finally, the empty data directory tree is created [Actually, it does that first].

## Improvements ##

* package the `fast_rolling.c` code so it can be installed with `pip`.
* add option to update an existing installation
* prompt for release version
* devise a method to update a release version with the current
  development code without having to create a new release first, but
  maybe that should be discouraged in order to be able to track
  results to specific code.
* containerize the code using `docker` or `singularity`
  [comparison article](https://pythonspeed.com/articles/containers-filesystem-data-processing/)  
  `singularity` might be preferable in our case, ref: Christopher
  Kelly, CSI seminar on containers for HPC applications.
  
-------------------------------------------------------------------------------

# run_forecast.sh #
This `bash` script automates the distributed running of the `~/release/code/scripted_processing` steps for historical/archived data.
It takes two command-line arguments:

* `DAY1` - the first date to be processed
* `NDAYS` - the number of consecutive days to process.

It defaults to the `config.conf` file in the current working directory, but can be overwritten with the `CONF` environment variable. The `SITE` defaults to `bnl`.
It actually modifies the config.conf file in place to insert the list of dates to process in the `days=` configuration line.

It does all of the following:

* copy the raw image data from the archive on `solar-db.bnl.gov`
* copy the raw GHI data if necessary.
* create separate log files for each processing step in `data/SITE/log/DAY1_NDAYS`
* run the processes
  * `preprocess`
  * `generate_stitch`
  * `extract_features`
  * `predict`
* copy the results (`stitch`,`feature`,`forecast`) and logs back to `solar-db.bnl.gov`

