# Solar Forecasting

## Quick Start

1. Install conda from the Anaconda [site](https://www.anaconda.com/products/individual)
2. Open a terminal and check that conda is available
    ```console
    conda --version
    ```
3. Create an environment for the project     
   ```console
   conda create --name ray_env python=3.9
   ```
4. Activate the environment    
   ```console
   conda activate ray_env
   ```
5. Upgrade pip    
   ```console
   pip install --upgrade pip
   ```
6. Install all dependencies  
    ```console
    pip install ray[default] netCDF4 pytz scipy numpy ephem cython scikit-image pvlib
	```
    (for viewer include pyqt5 pyqtgraph)
    
	(for GHI database include tzdata, sqlalchemy, pymysql)
    
    ```console
    conda install -c conda-forge pyfftw
    cd <SolarForecastingRoot>/rolling
    python setup.py install
    ```

7. Edit the file config.conf file with your paths and date selection

8. Optionally, start a Ray cluster.  (If not started, the local machine will run as auto-start as a local only head node.)

    Head Node:
    ```console
    ray start --head
    ```
    
    Client Nodes:
    ```console
    ray start --address='<head IP>:6379' --redis-password='<PW>'  (head node will provide redis password when started)
    ```
    
    Other useful options:
    ```console
        --dashboard-host=0.0.0.0  (head only, for outside non-local dashboard access)
        --object-store-memory=$((<bytes>))  (sets max object store memory in bytes, for example $((1024*1024*1024*64)) would set it to 64 GB)
        --system-config='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"<path>"}}"}'  (sets specific object store spill location)
        --num-cpus=<n>  (sets number of cores to use for node)
     ```
 
9. Now you'll be able to run the code:
   ```console
   python process_pipeline.py    
   python forecast_and_validate.py
   ```
   To run with a specific config file rather than the default config.conf, add it to the command line:
      ```console
   python process_pipeline.py alternate_config.conf
   ```
    
10. To fetch GHI data from database, update connection info in config then run:
    ```console
    python getFeatureGHI.py
    ```

