bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small


python bench_model.py -o data/models -r 10 -b 32 -i 256 -m bert-base-uncased bert-large-uncased distilbert-base-uncased google/mobilebert-uncased albert-base-v2 albert-large-v2 roberta-base roberta-large 



python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m openai-gpt gpt2 sshleifer/tiny-gpt2 distilgpt2 sshleifer/tiny-ctrl 2>&1 | tee data/bench-decoder.log

python bench_model.py -p -o data/models -r 10 -b 32 -i 256 -m facebook/bart-base facebook/bart-large sshleifer/distilbart-xsum-6-6 valhalla/distilbart-mnli-12-3 t5-small 2>&1 | tee data/bench-seq2seq.log

for b in `seq 32 -8 8` 1; do
  for i in `seq 512 -32 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml -o data/mlexp -r 5 -b ${b} -i ${i} -m "bert-base-uncased" "distilbert-base-uncased" "google/mobilebert-uncased" "roberta-base" -n 10000 2>&1 | tee data/ml-b${b}-i${i}.log
  done
done

for b in `seq 32 -8 8` 1; do
  for i in `seq 512 -32 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t model -o data/mlexp -r 5 -b ${b} -i ${i} -m "bert-base-uncased" "distilbert-base-uncased" "google/mobilebert-uncased" "roberta-base" -n 1000 2>&1 | tee data/nrg-b${b}-i${i}.log
  done
done

for b in `seq 32 -8 8` 1; do
  for i in `seq 512 -32 32`; do
    echo ${i},${b}=$(($i * $b))
    python run_level_exp.py -t ml -o data/mlexp -r 5 -b ${b} -i ${i} -m "bert-base-uncased" "distilbert-base-uncased" "google/mobilebert-uncased" "roberta-base" -n 10000 2>&1 | tee data/ml-b${b}-i${i}.log
  done
done
