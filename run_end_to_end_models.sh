#!/bin/sh
modelname=$1
for seqlen in 16 32 64 128 256 384 512
do
    for bs in 1 2 4 8 16 32 64
    do
        cmd="CUDA_VISIBLE_DEVICES=1 python end_to_end_pipeline.py -m $modelname --res_file $modelname-$seqlen-$bs-res.csv -j model_json_dumps -b $bs -i $seqlen 2> logs/$modelname-$seqlen-$bs-error.txt 1> logs/$modelname-$seqlen-$bs-output.txt"
        echo $cmd
        eval $cmd
    done
done
