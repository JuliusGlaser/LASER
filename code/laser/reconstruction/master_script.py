import yaml
from pathlib import Path
import subprocess
import time
import os
import shutil
from zipfile import ZipFile

def check_job_started(path, ending):
    # Specify the directory and file extension
    directory = Path(path)
    file_extension = '*.o' + ending  # Replace with the desired file extension
    print(file_extension)

    # Find files with the specific extension
    files_with_extension = list(directory.glob(file_extension))

    # Check if any file with the specified extension exists
    if files_with_extension:
        print('job started')
        return 1
    else:
        print(f"Job didn't start yet")
        return 0


# # Load and modify YAML config
def modify_yaml_config(config_path, key, value):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Modify the config as needed
    if key in config:
        config[key] = value

    # Save modified config
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)

yaml_config = 'decoder_recon_config.yaml'
# Path to your batch script
batch_script = 'cpu_batch_job.sh'

script_directory = os.path.dirname(os.path.abspath(__file__)) + '/'

# Command to execute (sbatch.tinyfat with the script)
command = ['sbatch.tinyfat', batch_script]

start = 0
step = 2
N_slices = 37


for i in range(start//step,N_slices//step+N_slices%step):
    print(i)
    modify_yaml_config(yaml_config, 'slice_index', i*step)
    # command.append(f'--slice_idx={i*step}')
    print('slice_index = ', i*step)
    # Run the command
    result = subprocess.run(command, capture_output=True, text=True)


    # Check for errors
    # if result.returncode == 0:
    #     print("Batch script submitted successfully!")
    #     print("Output:", result.stdout)
    # else:
    #     print("Error submitting batch script")
    #     print("Error message:", result.stderr)

    # Get the directory where the script is located
    
    print(script_directory)
    time.sleep(5)
    if check_job_started(script_directory, result.stdout.strip().split()[-4]) == 0:
        print('not enough capacity now')
        while(1):
            if check_job_started(script_directory, result.stdout.strip().split()[-4]) == 0:
                pass
            else:
                break
            time.sleep(20)
    else:
        print(f'script {i} started to run')
    time.sleep(4)







# # Run a script with the given config
# def run_script(script_name, config_path):
#     # Command to execute (sbatch.tinyfat with the script)
#     command = ['sbatch.tinyfat', script_name]

#     # Run the command
#     result = subprocess.run(command, capture_output=True, text=True)

#     # Check for errors
#     if result.returncode == 0:
#         print("Batch script submitted successfully!")
#         print("Output:", result.stdout)
#     else:
#         print("Error submitting batch script")
#         print("Error message:", result.stderr)

# # Zips the output folder after all jobs
# def zip_results(output_folder, zip_filename):
#     shutil.make_archive(zip_filename, 'zip', output_folder)
#     print(f"Results zipped to {zip_filename}.zip")

# # Main batch job logic
# def main(config_path, modifications_list, output_folder, num_batches, post_script):
#     for i in range(num_batches):
#         # Modify the YAML config for this batch
#         print(f"Batch {i + 1}/{num_batches}: Modifying config and running the main script.")
#         modify_yaml_config(config_path, modifications_list[i])

#         # Run the main script with the modified config
#         run_script('main_script.py', config_path)

#         # Sleep for 20 seconds before the next batch
#         time.sleep(20)

#     # Run the post script after all batch jobs
#     print("Running the post-processing script.")
#     run_script(post_script, config_path)

#     # Zip the result folder
#     zip_results(output_folder, 'batch_results')

# if __name__ == "__main__":
#     # Example usage
#     config_path = 'config.yaml'  # Path to your YAML config
#     output_folder = 'results/'   # Folder where batch job results are saved
#     num_batches = 5              # Number of batches to run

#     # Example list of modifications for each batch job
#     modifications_list = [
#         {'param1': 10, 'param2': 5},
#         {'param1': 20, 'param2': 15},
#         {'param1': 30, 'param2': 25},
#         {'param1': 40, 'param2': 35},
#         {'param1': 50, 'param2': 45},
#     ]

#     # Post-processing script
#     post_script = 'post_script.py'

#     # Run the batch job manager
#     main(config_path, modifications_list, output_folder, num_batches, post_script)