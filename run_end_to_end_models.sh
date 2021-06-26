#!/bin/sh

modelname=$1
device=$2
for seqlen in 32 64 96 128 160 192 224 256
do
    for bs in 8 16 24 32
    do
        cmd="CUDA_VISIBLE_DEVICES=$device python end_to_end_pipeline.py -m $modelname --res_file $modelname-$seqlen-$bs-res.csv -j model_json_dumps -b $bs -i $seqlen 2> logs/$modelname-$seqlen-$bs-error.txt"
        echo $cmd
        eval $cmd
    done
done
