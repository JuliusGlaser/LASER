## Bootstrapping 
This folder contains the scripts to run the MRtrix-based bootstrapping analysis.

### Usage
1) Install MRtrix3 and ensure it is available on your PATH.
2) Request the required data from the author.
3) Create the bootstrapping list:
```
python create_bootstrapping_list.py
```
4) Adjust `config.yaml` for your dataset and output locations.
5) Run the analysis:
```
python bootstrapping_analysis_mrtrix.py
```
