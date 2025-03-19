import argparse
import os
import platform

os_name = platform.system()

CURR_DIR = os.getcwd()
DATA_DIR = DIR = os.path.dirname(os.path.realpath(__file__))
print(CURR_DIR)
print(DATA_DIR)

# %%
files_list = ['1.0mm_126-dir_R3x3_coils.h5',
              '1.0mm_126-dir_R3x3_dvs.h5',
              '1.0mm_126-dir_R3x3_kdat_slice_000.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_001.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_002.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_003.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_004.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_005.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_006.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_007.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_008.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_009.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_010.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_011.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_012.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_013.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_014.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_015.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_016.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_017.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_018.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_019.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_020.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_021.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_022.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_023.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_024.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_025.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_026.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_027.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_028.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_029.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_030.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_031.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_032.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_033.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_034.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_035.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_036.h5',
            #   '1.0mm_126-dir_R3x3_kdat_slice_037.h5',
              '1.0mm_126-dir_R3x3_refs.h5']

# %%
parser = argparse.ArgumentParser(description='load NAViEPI data from Zenodo.')

# here multiple files should be separated with space
parser.add_argument('--file', default=None, type=str,
                    help='name of the file to be downloaded. Note: \
                        multiple files should be separated with space. \
                            Default is None, i.e. download all files.')

# args = parser.parse_args(args=None if os.sys.argv[1:] else ['--help'])
args = parser.parse_args()

# %% download data
if bool(args.file and not args.file.isspace()):  # not blank
    print('> user provided file: ', args.file)
    files_list = args.file.split()
else:
    print('> download all data (slow!)')

# download
for f in files_list:

    if os.path.exists(DATA_DIR + '/' + f):
        print(f'The file {f} exists.')
    else:
        if os_name == 'Linux':
            os.system('wget -P ' + DATA_DIR + ' -q https://zenodo.org/records/13171692/files/' + f)
        elif os_name == 'Windows':
            url = 'https://zenodo.org/records/13171692/files/' + f
            os.system(f'powershell -Command "Invoke-WebRequest -Uri {url} -OutFile {DATA_DIR}{os.sep}{f}"')


# check
if os_name == 'Linux':
    os.chdir(DATA_DIR)

    for f in files_list:
        os.system('cat md5sum.txt | grep ' + f + ' | md5sum -c --ignore-missing')

    os.chdir(CURR_DIR)