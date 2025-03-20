## Downloading the data

To download all data, you need to run:
```bash
python load.py
```
As the complete data has a size of about 50 Gb, we recommend to just load the necessary files, being the coil sensetivities and dvs file, and a certain amount of slices:
 ```bash
python load.py --file 1.0mm_126-dir_R3x3_coils.h5 1.0mm_126-dir_R3x3_dvs.h5
python load.py --file 1.0mm_126-dir_R3x3_kdat_slice_000.h5 1.0mm_126-dir_R3x3_kdat_slice_001.h5 #...
```