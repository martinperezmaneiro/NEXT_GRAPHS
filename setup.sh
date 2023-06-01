#!/usr/bin/bash

### Finally found a way that from any machine you run source setup.sh and it will create the main directory (in this case GNN) without problem
### because it uses the dirname "$0" to get the dir to the setup.sh file from where we are running the source setup.sh, then goes to this
### directory, does a pwd and this is what we get, the full path to the repository :)

export GNN=$( cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 ; pwd -P )/
export PYTHONPATH=$GNN:$PYTHONPATH
export PATH=$GNN/bin:$PATH
export PATH=$GNN/scripts:$PATH