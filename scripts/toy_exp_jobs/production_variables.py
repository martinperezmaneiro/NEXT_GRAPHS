import os
import glob
import re


#Directory that will contain the created jobs/configs and the output files
basedir = os.path.expandvars("/mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/graph_toy_experiment/")
total_events = 1000000
events_per_file = 1000
nfiles = int(total_events / events_per_file)

out_filename = "toy_exp_{num}.pt"

#number of jobs to launch (max is 30 in ft3, but you can add any number of tasks per job
#while it doesn't pass the time per job)
queue_limit   = 30
tasks_per_job = 20

#directory of the job and config templates, write / at the end
taskTemp_dir   = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")
configTemp_dir = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")
jobTemp_dir    = os.path.expandvars("/home/usc/ie/mpm/NEXT_graphs/templates/")

configTemp_filename = "toy_expTemplate.conf"
taskTemp_filename   = "taskTemplate.sh"
jobTemp_filename    = "jobTemplate_ft3.sh"

#path of the script to run
scriptdir = "/home/usc/ie/mpm/NEXT_graphs/scripts/create_toy_exp.py"

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


#Specifications for the tasks in each job
task_params = "srun --ntasks 1 --exclusive --cpus-per-task 1 "

##############
# JOB LAUNCH
##############

#commands for CESGA
queue_state_command = "squeue -r |grep usciempm |wc -l"
joblaunch_command   = "sbatch {job_filename}"
jobtime             = "5:59:00"
