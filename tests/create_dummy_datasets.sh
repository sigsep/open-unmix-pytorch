#!/usr/bin/env bash

# Set the number of files to generate
NBFILES=4
BASEDIR=TrackfolderDataset
subsets=(
    train
    valid
)
for subset in "${subsets[@]}"; do
    for k in $(seq 1 4); do
        path=$BASEDIR/$subset/$k
        mkdir -p $path 
        for i in $(seq 1 $NBFILES); do
            sox -n -r 8000 -b 16 $path/$i.wav synth "0:3" whitenoise vol 0.5 fade q 1 "0:3" 1
        done
    done
done
