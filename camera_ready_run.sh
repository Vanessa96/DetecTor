#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2" "openai-gpt" "gpt2")

start_T=$(date +"%T")
echo "Start time : $start_T"
r=1

for b in 24 32; do
  for i in 32 128; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -le -t ml-np -eo data2/soft-ml-qpc-${model}-b${b}-i${i} -o data2/soft-ml-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data2/logs2/soft-ml-np-qpc-b${b}-i${i}.log
      python run_level_exp.py -le -t ml -eo data2/soft-ml-qpc-${model}-b${b}-i${i} -o data2/soft-ml-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data2/logs2/soft-ml-qpc-b${b}-i${i}.log
      python run_level_exp.py -le -t module -eo data2/soft-module-qpc-${model}-b${b}-i${i} -o data2/soft-module-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data2/logs2/soft-module-qpc-b${b}-i${i}.log
      # python run_level_exp.py -le -t model -eo data2/soft-model-qpc-${model}-b${b}-i${i} -o data2/soft-model-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data2/logs2/soft-model-qpc-b${b}-i${i}.log
    done
  done
done

mid_T=$(date +"%T")
echo "Mid time : $mid_T, started at $start_T"

for b in 24 32; do
  for i in 32 128; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python calc_soft_energy.py -sw -e data2/soft-ml-qpc-${model}-b${b}-i${i} -p data2/soft-ml-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -t ml-np 1>>cmr_ml2.log
      python calc_soft_energy.py -sw -e data2/soft-ml-qpc-${model}-b${b}-i${i} -p data2/soft-ml-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -t ml 1>>cmr_ml2.log
      python calc_soft_energy.py -sw -e data2/soft-module-qpc-${model}-b${b}-i${i} -p data2/soft-module-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -t module 1>>cmr_module2.log
      # python calc_soft_energy.py -sw -e data2/soft-model-qpc-${model}-b${b}-i${i} -p data2/soft-model-exp-qpc -r ${r} -b ${b} -i ${i} -m ${model} -t model
    done
  done
done

end_T=$(date +"%T")
echo "End time : $end_T, started at $start_T, mid time at $mid_T"
