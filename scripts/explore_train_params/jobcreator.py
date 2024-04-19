
"""
Assumes tasks are already created because each job will take several tasks
"""

import os
import glob
from math import ceil
from taskcreator import basedir

task_dir = basedir + 'neural_network/*/'
tasks_per_job = 10

jobs_dir = basedir + 'neural_network/jobs/'
logs_dir = basedir + 'neural_network/logs/'

job_temp = '/home/usc/ie/mpm/NEXT_graphs/templates/jobTemplate_ft3.sh'
task_params = "srun --ntasks 1 --exclusive --cpus-per-task 1 "
jobtime = "6:00:00"


if __name__ == "__main__":
    os.makedirs(jobs_dir)
    os.makedirs(logs_dir)
    task_filenames = sorted(glob.glob(os.path.expandvars(task_dir + 'task.sh')))

    ntasks = len(task_filenames)
    nbatches = ceil(ntasks/tasks_per_job)

    print(f"Creating jobs")

    ####### CREATING JOBS (for each job, we will have some tasks) #######
    #take the template path
    #open the template to use it
    job_file = open(job_temp).read()

    #Write jobs
    for batch in range(0, nbatches):
        tasks_in_batch = task_filenames[batch*tasks_per_job:(batch+1)*tasks_per_job]

        #Write a command line for each task inside the job
        task_commands = ""
        for task in tasks_in_batch:
            task_commands += task_params + f"{task} &\n"

        #Write the task commands to the file
        #Create file to write
        job = jobs_dir + f"job_{batch + 1}.sh"
        with open(job, "x") as job_write:
            job_write.write(job_file.format(jobname = str(batch + 1),
                                            logfilename = logs_dir + str(batch + 1) + ".log",
                                            errfilename = logs_dir + str(batch + 1) + ".err",
                                            tasks_per_job = len(tasks_in_batch),
                                            jobtime = jobtime,
                                            tasks = task_commands))

    print(f"{nbatches} jobs created")
