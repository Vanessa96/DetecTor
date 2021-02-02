bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small


python bench_model.py -o data/models -r 10 -b 32 -i 256 -m bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large



python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl 2>&1 | tee data/bench-decoder.log

python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small 2>&1 | tee data/bench-seq2seq.log

mkdir -p data/small-exp
rprof/rprof 170 data/small-exp/ml-exp-jpc-res.csv 100000

for b in `seq 32 -4 4`; do
  for i in `seq 512 -16 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml -o data/ml-exp-jpc -r 5 -b ${b} -i ${i} -m "distilbert-base-uncased" "roberta-base" "google/mobilebert-uncased" "bert-base-uncased"  -n 10000 2>&1 | tee data/ml-logs/ml-jpc-b${b}-i${i}.log
  done
done

for b in `seq 32 -8 8` 1; do
  for i in `seq 256 -32 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t model -o data/model-exp2 -r 5 -b ${b} -i ${i} -m "google/mobilebert-uncased" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" -n 100 2>&1 | tee data/nrg-b${b}-i${i}.log
  done
done

for b in `seq 8 8 32`; do
  for i in `seq 32 32 512`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t module -o data/module-exp -r 10 -b ${b} -i ${i} -m "google/mobilebert-uncased" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" -n 1000 2>&1 | tee data/module-logs/module-b${b}-i${i}.log
  done
done

b=2
i=6
python run_level_exp.py -t ml-np -o data/ml-np-exp -r 2 -b ${b} -i ${i} -m "google/mobilebert-uncased" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"  -n 100 2>&1 | tee data/ml-np-b${b}-i${i}.log


for b in `seq 32 -4 4`; do
  for i in `seq 512 -16 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml -o data/ml-exp-jpc -r 5 -b ${b} -i ${i} -m "distilbert-base-uncased" "roberta-base" "google/mobilebert-uncased" "bert-base-uncased"  -n 10000 2>&1 | tee data/ml-logs/ml-jpc-b${b}-i${i}.log
  done
done

declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base")

declare -a models=("bert-large-uncased" "google/mobilebert-uncased" "albert-base-v2" "albert-large-v2" "roberta-large" "openai-gpt" "gpt2" "sshleifer/tiny-gpt2" "distilgpt2" "sshleifer/tiny-ctrl" "facebook/bart-base" "facebook/bart-large" "sshleifer/distilbart-xsum-6-6" "valhalla/distilbart-mnli-12-3" "t5-small")

declare -a models=("squeezebert/squeezebert-uncased" "huawei-noah/TinyBERT_General_4L_312D" "xlnet-base-cased" "funnel-transformer/small" "google/electra-small-discriminator" "google/mobilebert-uncased" "facebook/bart-base" "facebook/bart-large" "sshleifer/distilbart-xsum-6-6" "valhalla/distilbart-mnli-12-3" "t5-small")


r=1
b=8
i=128
for model in "${models[@]}"; do
  echo ${i},${b}=$(($i * $b))
  python run_level_exp.py -t ml-np -o data/ml-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/ml-np-jpc-${model}-b${b}-i${i}.log
  python run_level_exp.py -t ml -o data/ml-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/ml-jpc-${model}-b${b}-i${i}.log
  python run_level_exp.py -t module -o data/module-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/module-jpc-${model}-b${b}-i${i}.log
  python run_level_exp.py -t model -o data/model-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/model-jpc-${model}-b${b}-i${i}.log
done

for b in `seq 4 4 32`; do
  for i in `seq 8 8 256`; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -t ml-np -o data/ml-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/ml-np-jpc-b${b}-i${i}.log
      python run_level_exp.py -t ml -o data/ml-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/ml-jpc-b${b}-i${i}.log
      python run_level_exp.py -t module -o data/module-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/module-b${b}-i${i}.log
      python run_level_exp.py -t model -o data/model-exp-jpc -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/model-b${b}-i${i}.log
    done
  done
done

python run_level_exp.py -t model -o data/model-exp-jpc -r 5 -b ${b} -i ${i} -m ${models} -n 100

declare -a models=("distilbert-base-uncased" "roberta-base")
for model in "${models[@]}"; do
for b in `seq 8 8 32`; do
  for i in `seq 32 32 512`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml-np -o data/ml-np2 -r 5 -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/ml-np-logs/ml-np-b${b}-i${i}.log
  done
done
done

"google/mobilebert-uncased" "huawei-noah/TinyBERT_General_4L_312D" "google/electra-small-discriminator" "albert-base-v2" "openai-gpt" "gpt2" "sshleifer/tiny-gpt2" "distilgpt2" "sshleifer/tiny-ctrl"

declare -a models=("facebook/bart-base" "squeezebert/squeezebert-uncased" "sshleifer/distilbart-xsum-6-6")

r=1
b=8
i=128
machine="jpc"
for model in "${models[@]}"; do
  python run_level_exp.py -t ml-np -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/ml-np-${machine}-${model//\//_}-b${b}-i${i}.log
  python run_level_exp.py -t ml -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/ml-${machine}-${model//\//_}-b${b}-i${i}.log
  python run_level_exp.py -t module -o data/module-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/module-${machine}-${model//\//_}-b${b}-i${i}.log
  python run_level_exp.py -t model -o data/model-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/model-${machine}-${model//\//_}-b${b}-i${i}.log
done

declare -a models=("huawei-noah/TinyBERT_General_4L_312D" "google/electra-small-discriminator" "albert-base-v2" "sshleifer/tiny-gpt2")

declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "google/mobilebert-uncased" "sshleifer/tiny-ctrl")
r=1
machine="qpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -t ml-np -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/ml-np-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -t ml -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/ml-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -t module -o data/module-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/module-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -t model -o data/model-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/model-${machine}-${model//\//_}-b${b}-i${i}.log
    done
  done
done

for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    echo echo ${i},${b}=$(($i * $b))
  done
done | wc -l

  for b in `seq 4 4 32`; do
    for i in `seq 8 8 384`; do

machine="jpc"
data_dir="data/${machine}_8"
energy_file="${data_dir}/all-${machine}-energy.csv"
res_file="${data_dir}/all-${machine}-res.csv"
declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "sshleifer/tiny-ctrl")

for level in module ml model ml-np; do
  echo ${level//-np/}-exp-${machine}
  python gen_feature.py -o ${data_dir} -t ${level} -e ${level//-np/}-exp-${machine} -ef ${energy_file} -rf ${res_file} -m ${models[@]}  -r 1 --input_start 32 --seq_step 32 --input_length 257 --batch_start 8 --batch_step 8 --batch_size 33
done


scp -i .ssh/id_rsa_epi-jpc pi@130.245.145.100:~/emonpi/firmware/soft-jpc-energy.csv .

declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2")
r=5
machine="jpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -t model -le -eo data/soft-${machine}-${model//\//_}-b${b}-i${i} -o data/soft-model-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/soft-model-${machine}-${model//\//_}-b${b}-i${i}.log
    done
  done
done

declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "google/mobilebert-uncased" "sshleifer/tiny-ctrl")
r=1
b=8
i=32
machine="qpc"
for model in "${models[@]}"; do
  echo ${i},${b}=$(($i * $b))
  python calc_soft_energy.py -e data/soft/soft-model-exp-${machine}-${model} -p data/soft/soft-model-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model}
done

data_dir="data/exp"
energy_file="${data_dir}/all-exp-qpc-energy.csv"
res_file="${data_dir}/all-exp-qpc-res.csv"
declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "google/mobilebert-uncased" "sshleifer/tiny-ctrl")

python gen_feature.py -o ${data_dir} -t ${level} -e ${level//-np/}-exp-qpc -ef ${energy_file} -rf ${res_file} -m ${models[@]}  -r 1 --input_start 32 --seq_step 32 --input_length 257 --batch_start 8 --batch_step 8 --batch_size 33

declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2")
r=5
machine="jpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python calc_soft_energy.py -e data/jpc-soft/soft-${machine}-${model//\//_}-b${b}-i${i} -p data/jpc-soft/soft-model-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -ef data/jpc-soft/soft-jpc-energy.csv 2>&1 | tee data/jpc-soft/soft-model-${machine}-${model//\//_}-b${b}-i${i}.log
    done
  done
done


r=1
machine="qpc"
for model in "${models[@]}"; do
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -t ml -o data/soft-model-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10
    done
  done
done

declare -a models=("prajjwal1/bert-tiny")


declare -a models=("prajjwal1/bert-tiny" "bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "google/mobilebert-uncased" "sshleifer/tiny-ctrl")

declare -a models=("huawei-noah/TinyBERT_General_4L_312D" "google/electra-small-discriminator" "sshleifer/tiny-gpt2")
for model in "${models[@]}"; do
 python visualise_model_as_graph.py --model-name ${model} --out-file data/viz/${model//\//_}.viz
 dot -Tpdf data/viz/${model//\//_}.viz -o data/viz/${model//\//_}.pdf
done

r=1
machine="jpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for a in 2 4 8 12; do
      h=$((a*64))
      for l in $(seq 2 2 12); do
        if [ $a -eq 12 ] && [ $l -eq 12 ]; then
          continue
        fi
        model="google/bert_uncased_L-${l}_H-${h}_A-${a}"
        echo "running ${model} for i${i},b${b}..."
        python run_level_exp.py -t model -o data/model-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/model-${machine}-${model//\//_}-b${b}-i${i}.log
        python run_level_exp.py -t module -o data/module-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/module-${machine}-${model//\//_}-b${b}-i${i}.log
      done
    done
  done
done

r=1
machine="jpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for a in 2 4 8 12; do
      h=$((a*64))
      for l in $(seq 2 2 12); do
        if [ $a -eq 12 ] && [ $l -eq 12 ]; then
          continue
        fi
        model="google/bert_uncased_L-${l}_H-${h}_A-${a}"
        echo "running ${model} for i${i},b${b}..."
        python run_level_exp.py -t ml-np -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/ml-np-${machine}-${model//\//_}-b${b}-i${i}.log
        python run_level_exp.py -t ml -o data/ml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/ml-${machine}-${model//\//_}-b${b}-i${i}.log
      done
    done
  done
done


  for a in 2 4 8 12; do
    h=$((a*64))
    for l in $(seq 2 2 12); do
    if [ $a -eq 12 ] && [ $l -eq 12 ]; then
      continue
    fi
      echo $l,$h
    done
  done

nvprof --metrics sm_efficiency,shared_efficiency,dram_utilization --continuous-sampling-interval 100 --profile-from-start off --csv --log-file output-bert-tiny-p.csv python bench_model.py -p -o data/models -r 10 -b 32 -i 120 -m "prajjwal1/bert-tiny"

nvprof --metrics flop_dp_efficiency,flop_sp_efficiency,gld_efficiency,gld_throughput,gld_transactions,gst_efficiency,gst_throughput,gst_transactions,inst_compute_ld_st,inst_control,inst_issued,inst_executed,sm_efficiency,dram_read_transactions,dram_write_transactions --continuous-sampling-interval 10 --profile-from-start off --csv --log-file output-bert-tiny-p.csv

nvprof -f --analysis-metrics --export-profile bert-tiny.db --log-file output-bert-tiny-p.csv python bench_model.py -p -o data/models -r 10 -b 32 -i 120 -m "prajjwal1/bert-tiny"

TASK_NAME=mrpc
TASK_NAME=sst2
TASK_NAME=mnli
for TASK_NAME in mnli sst2 mrpc; do
  for a in 2 4 8 12; do
    h=$((a*64))
    for l in $(seq 2 2 12); do
      model="google/bert_uncased_L-${l}_H-${h}_A-${a}"
      echo "finetuning ${model} for $TASK_NAME"
      python run_glue.py \
        --model_name_or_path ${model} \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length 128 \
        --per_device_train_batch_size 32 \
        --learning_rate 3e-5 \
        --num_train_epochs 3 \
        --save_total_limit 1 \
        --output_dir data/$TASK_NAME/${model} 2>&1 | tee data/logs/$TASK_NAME-${model//\//_}.log
    done
  done
done

for TASK_NAME in mnli sst2; do
echo $TASK_NAME
done


declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "sshleifer/tiny-ctrl")
r=1
machine="qpc"
for b in `seq 8 8 32`; do
  for i in `seq 32 32 256`; do
    for model in "${models[@]}"; do
      echo ${i},${b}=$(($i * $b))
      python run_level_exp.py -mg -t ml-np -o data/pml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 100 2>&1 | tee data/logs/pml-np-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -mg -t ml -o data/pml-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model}  -n 100 2>&1 | tee data/logs/pml-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -mg -t module -o data/pmodule-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 50 2>&1 | tee data/logs/pmodule-${machine}-${model//\//_}-b${b}-i${i}.log
      python run_level_exp.py -mg -t model -o data/pmodel-exp-${machine} -r ${r} -b ${b} -i ${i} -m ${model} -n 10 2>&1 | tee data/logs/pmodel-${machine}-${model//\//_}-b${b}-i${i}.log
    done
  done
done

models=()
for a in 2 4 8 12; do
  h=$((a*64))
  for l in $(seq 2 2 12); do
    models+=("google/bert_uncased_L-${l}_H-${h}_A-${a}")
  done
done
echo $models


machine="jpc"
data_dir="data/bert24_${machine}"
energy_file="${data_dir}/bert24-${machine}-energy.csv"
res_file="${data_dir}/bert24-${machine}-res.csv"

for level in ml ml-np module model; do
  echo ${level//-np/}-exp-${machine}
  python gen_feature.py -o ${data_dir} -t ${level} -e ${level//-np/}-exp-${machine} -ef ${energy_file} -rf ${res_file} -m ${models[@]}  -r 1 --input_start 32 --seq_step 32 --input_length 257 --batch_start 8 --batch_step 8 --batch_size 33
done