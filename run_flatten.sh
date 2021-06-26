search_dir=e2e_cache
for entry in "$search_dir"/*
do
    echo $entry
    python convert_pkl_to_json.py --fname $entry -o model_json_dumps/
done