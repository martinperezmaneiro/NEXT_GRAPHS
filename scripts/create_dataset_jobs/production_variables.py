import os
import glob
import re



#Directory that will contain the created jobs/configs and the output files
basedir = os.path.expandvars("/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/13bar/0nubb/graph_nn/")
#directory of the input files to be processed, remember to write / at the end
indir   = os.path.expandvars("/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/13bar/0nubb/label/prod/")

tag = "0nubb"
#number of jobs to launch (max is 30 in ft3, but you can add any number of tasks per job
#while it doesn't pass the time per job)
queue_limit   = 30
tasks_per_job = 40

#path of the script to run
scriptdir = "/home/usc/ie/mpm/NEXT_graphs/scripts/create_graphs_from_data.py"

#directory of the job and config templates, write / at the end
taskTemp_dir   = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")
configTemp_dir = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")
jobTemp_dir    = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")


#name for the in/out files and names of the job/config templates
in_filename         = "nexus_label_{num}_{tag}.h5"
out_filename        = "nexus_graph_nn_{num}_{tag}.pt"

configTemp_filename = "create_datasetTemplate.conf"
taskTemp_filename   = "taskTemplate.sh"
jobTemp_filename    = "jobTemplate_ft3.sh"

#function to get the num of each file
get_file_number = lambda filename: int(filename.split("/")[-1].split("_")[-2])

def check_filename_structure(filename):
    name  = filename.split("/")[-1]
    # check length
    assert len(name.split("_")) == len(in_filename.split("_"))
    # check file number
    assert name.split("_")[-2].isdigit()
    # check tag
    assert name.split("_")[-1] == tag + ".h5"


#checks if a directory exists, if not, it creates it
def checkmakedir(path):
	if os.path.isdir(path):
		print('hey, directory already exists!:\n' + path)
	else:
		os.makedirs(path)
		print('creating directory...\n' + path)

#this function creates the output tree of directories (all the jobs and configs, the data
#production and the logs
def create_out_dirs():
    proddir = basedir + "/prod/"
    taskdir = basedir + "/tasks/"
    jobsdir = basedir + "/jobs/"
    confdir = basedir + "/config/"
    logsdir = basedir + "/logs/"
    checkmakedir(proddir)
    checkmakedir(taskdir)
    checkmakedir(jobsdir)
    checkmakedir(confdir)
    checkmakedir(logsdir)

    return proddir, taskdir, jobsdir, confdir, logsdir

proddir, taskdir, jobsdir, confdir, logsdir = create_out_dirs()

##############
# INPUT FILES
##############

#takes all the .h5 files in the specified indir. we will make a loop on them, but as I
#want to grab all the files with the same cut, in the job creator script I will assure that
#once one cut job/config is created, no other configs are created
files_in = glob.glob(indir + "/*.h5") #para que me haga todo poner *.h5 al final
for file_in in files_in: check_filename_structure(file_in)
#sorts all the files by number
files_in = sorted(files_in, key = get_file_number)

#Specifications for the tasks in each job
task_params = "srun --ntasks 1 --exclusive --cpus-per-task 1 "

##############
# JOB LAUNCH
##############

#commands for CESGA
queue_state_command = "squeue -r |grep usciempm |wc -l"
joblaunch_command   = "sbatch {job_filename}"
jobtime             = "5:59:00"
