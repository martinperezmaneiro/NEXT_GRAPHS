#!/bin/bash

#SBATCH --job-name gnn_dataset
#SBATCH --output   /home/usc/ie/mpm/NEXT_graphs/scripts/create_graph_dataset/gnn_dataset.out
#SBATCH --error    /home/usc/ie/mpm/NEXT_graphs/scripts/create_graph_dataset/gnn_dataset.err
#SBATCH --ntasks   1
#SBATCH --time     6:00:00
#SBATCH --cpus-per-task 100
#SBATCH --mem-per-cpu 4G

source $STORE/ic_nn_setup.sh

srun --ntasks 1 --exclusive --cpus-per-task 100 /home/usc/ie/mpm/NEXT_graphs/scripts/create_graph_dataset/task_create_dataset.sh &

wait