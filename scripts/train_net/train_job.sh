#!/bin/bash
#
#
# TEMPLATE FOR IC PRODUCTIONS AT FT3
#
#SBATCH --job-name gnn_train
#SBATCH --output   /mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/13bar/train/logs/job_log.out
#SBATCH --error    /mnt/lustre/scratch/nlsas/home/usc/ie/mpm/NEXT100/data/pressure_topology/13bar/train/logs/job_err.err
#SBATCH --ntasks   1
#SBATCH --time     6:00:00
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 20G


#################################
#########  JOB CORE ############
################################
# set up IC in the CESGA machine
source $STORE/ic_nn_setup.sh
source $HOME/NEXT_graphs/setup.sh
pwd

srun --ntasks 1 --exclusive --cpus-per-task 1 /home/usc/ie/mpm/NEXT_graphs/scripts/train_net/train_task.sh
wait
