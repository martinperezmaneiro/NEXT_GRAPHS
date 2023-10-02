#!/bin/bash
#
#
# TEMPLATE FOR IC PRODUCTIONS AT FT3
#
#SBATCH --job-name {jobname}
#SBATCH --output   {logfilename}
#SBATCH --error    {errfilename}
#SBATCH --ntasks   {tasks_per_job}
#SBATCH --time     {jobtime}
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 3G
# these are the ones used by gonzalo (above)
##SBATCH --qos shared_short
##SBATCH --partition shared
##SBATCH --qos amd-shared
##SBATCH --partition amd-shared
##SBATCH --qos default
##SBATCH --partition thin-shared
##SBATCH --qos cl-intel-shared
##SBATCH --partition cl-intel-shared
##SBATCH -n 1
##SBATCH -N 1


#################################
#########  JOB CORE ############
################################
# set up IC in the CESGA machine
source $STORE/ic_nn_setup.sh
source $HOME/NEXT_graphs/setup.sh
pwd

{tasks}
wait
