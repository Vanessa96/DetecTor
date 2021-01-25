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
machine="jpc"
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


data_dir="data/exp/qpc-new"
energy_file="${data_dir}/all-exp-qpc-energy.csv"
res_file="${data_dir}/all-exp-qpc-res.csv"
declare -a models=("bert-base-uncased" "distilbert-base-uncased" "roberta-base" "distilgpt2"  "openai-gpt" "gpt2" "google/mobilebert-uncased" "sshleifer/tiny-ctrl")

for level in module ml model ml-np; do
  echo ${level//-np/}-exp-qpc
  python gen_feature.py -o ${data_dir} -t ${level} -e ${level//-np/}-exp-qpc -ef ${energy_file} -rf ${res_file} -m ${models[@]}  -r 1 --input_start 32 --seq_step 32 --input_length 257 --batch_start 8 --batch_step 8 --batch_size 33
done


scp -i .ssh/id_rsa_epi-jpc pi@130.245.145.100:~/emonpi/firmware/all-exp-jpc-energy.csv .

