bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small


python bench_model.py -o data/models -r 10 -b 32 -i 256 -m bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large 



python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl 2>&1 | tee data/bench-decoder.log

python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small 2>&1 | tee data/bench-seq2seq.log

mkdir -p data/small-exp
rprof/rprof 170 data/small-exp/ml-exp-jpc-res.csv 100000

for b in `seq 32 -4 4`; do
  for i in `seq 512 -16 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml -o data/ml-exp-jpc -r 10 -b ${b} -i ${i} -m "distilbert-base-uncased" "roberta-base" "google/mobilebert-uncased" "bert-base-uncased"  -n 10000 2>&1 | tee data/ml-logs/ml-jpc-b${b}-i${i}.log
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

b=32
i=128
python run_level_exp.py -t module -o data/module-exp -r 2 -b ${b} -i ${i} -m "google/mobilebert-uncased" "bert-base-uncased" "distilbert-base-uncased" "roberta-base"  -n 100 2>&1 | tee data/module-logs/module-b${b}-i${i}.log

