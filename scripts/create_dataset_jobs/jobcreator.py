
"""
Assumes tasks are already created because each job will take several tasks
"""

import os
import re
import glob
from math import ceil
from production_variables import *


if __name__ == "__main__":
    get_file_number = lambda filename: int(filename.split("_")[-1].split(".")[0])

    task_filenames = sorted(glob.glob(os.path.expandvars(taskdir + "task_*.sh")),
                            key = get_file_number)

    ntasks = len(task_filenames)
    nbatches = ceil(ntasks/tasks_per_job)

    print(f"Creating {tag} labelling jobs")

    ####### CREATING JOBS (for each job, we will have some tasks) #######
    #take the template path
    job_temp = jobTemp_dir + jobTemp_filename
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
        job = jobsdir + f"job_{batch + 1}.sh"
        with open(job, "x") as job_write:
            job_write.write(job_file.format(jobname = str(batch + 1) + "_" + tag,
                                            logfilename = logsdir + str(batch + 1) + ".log",
                                            errfilename = logsdir + str(batch + 1) + ".err",
                                            tasks_per_job = len(tasks_in_batch),
                                            jobtime = jobtime,
                                            tasks = task_commands))

    print(f"{nbatches} jobs created")
