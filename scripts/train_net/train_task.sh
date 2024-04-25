#!/bin/bash

start=`date +%s`

python /home/usc/ie/mpm/NEXT_graphs/scripts/main.py -a train -conf /home/usc/ie/mpm/NEXT_graphs/scripts/train_net/GCNClass.conf

end=`date +%s`
let deltatime=end-start
let hours=deltatime/3600
let minutes=(deltatime/60)%60
let seconds=deltatime%60
printf "Time spent: %d:%02d:%02d\n" $hours $minutes $seconds
