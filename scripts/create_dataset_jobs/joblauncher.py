import os
import re
import glob
import subprocess
from time import sleep

from production_variables import *


def check_jobs(cmd, nmin=10, wait=1):
    j = nmin
    while (j>nmin-1):
        #Entiendo que corre en cmd el comando q se manda, y guarda el otput en j
        #Que el output será le numero de jobs que hay corriendo
        j = subprocess.run(cmd, shell=True, capture_output=True)
        j = int(j.stdout)
        #Entonces cuando pasa, este wait entiendo que hace que salga del bucle
        #para mandar un job en el bucle de joblaunching; luego volverá aquí
        sleep(wait)
        if (j == nmin): sleep(10*wait)

############
# LAUNCHER
############


jobs = glob.glob(os.path.expandvars(jobsdir + "/job_*.sh"))


if __name__ == "__main__":
    get_file_number = lambda filename: int(filename.split("_")[-1].split(".")[0])

    jobs = sorted(jobs, key = get_file_number)

    for job in jobs:
        check_jobs(queue_state_command, nmin = queue_limit)
        #launch job
        cmd = joblaunch_command.format(job_filename = job)
        print("Launching job", job)
        subprocess.run(cmd, shell = True)
