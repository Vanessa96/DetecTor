#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.float_format', '{:.6f}'.format)


def is_float(x):
    try:
        float(x)
    except ValueError:
        return False
    return True


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs
    input_length = args.input_length
    batch_size = args.batch_size
    batch_step = args.batch_step
    seq_step = args.seq_step
    model_name = 'bert-base-uncased'
    level = 'level'
    feature_file = out_dir / f'{model_name}_features.csv'

    res = pd.read_csv(out_dir / f'res-{model_name}.csv')
    energy = pd.read_csv(out_dir / f'energy-{model_name}.csv',
                         error_bad_lines=False, usecols=[0, 2])
    energy = energy[energy['value'].apply(lambda x: is_float(x))]
    energy = energy[energy['timestamp'].apply(lambda x: is_float(x))]

    energy['value'] = energy['value'].astype(float).div(100)
    energy['timestamp'] = energy['timestamp'].astype(float)

    res_np = res.to_numpy()
    res_t = res_np[:, 0]
    energy_np = energy.to_numpy()
    energy_t = energy_np[:, 0]
    res_names = ['cpu', 'mem', 'gpu', 'gpu_mem']

    feature_names = ['batch_size', 'seq_len', 'flops',
                     'mem_bytes'] + res_names + [f'{k}_std'
                                                 for k in res_names] + \
                    ['times_mean', 'times_std',
                     'gpu_power_mean', 'gpu_power_std',
                     'energy_mean', 'energy_std',
                     'level_name', 'model_name']
    feature_values = {k: [] for k in feature_names}

    for bs in list(range(2, batch_size, batch_step)) + [1]:
        for seq_len in range(16, input_length, seq_step):
            filename = f'{model_name}_{level}_r{runs}_b{bs}_i{seq_len}.json'
            prof_file = Path(out_dir) / 'mlexp' / filename
            if not prof_file.exists():
                continue
            with open(prof_file) as f:
                prof_info = json.load(f)

            for prof_item in prof_info:
                gpu_power_runs = []
                energy_runs = []
                times_runs = []
                res_runs = {k: [] for k in res_names}
                repeats = prof_item['repeats']
                for r in range(1, runs + 1):
                    start_r = prof_item[f'start_{r}']
                    end_r = prof_item[f'end_{r}']
                    times_runs.append(end_r - start_r)

                    res_s = bisect.bisect_right(res_t, start_r)
                    res_e = bisect.bisect_right(res_t, end_r)
                    res_r = res[res_s:res_e]
                    for rn in res_names:
                        res_runs[rn].append(res_r[rn].mean())
                    gpu_power_r = res_r['gpu_power'].sum()
                    gpu_power_runs.append(gpu_power_r)

                    e_s = bisect.bisect_right(energy_t, start_r)
                    e_e = bisect.bisect_right(energy_t, end_r)
                    energy_r = energy[e_s:e_e]['value'].sum().div(repeats)
                    energy_runs.append(energy_r)

                times_mean = np.mean(times_runs)
                times_std = np.std(times_runs) / times_mean * 100
                gpu_power_mean = np.mean(gpu_power_runs)
                gpu_power_std = np.std(gpu_power_runs) / gpu_power_mean * 100
                energy_mean = np.mean(energy_runs)
                energy_std = np.std(energy_runs) / energy_mean * 100
                for rn in res_names:
                    feature_values[rn].append(np.mean(res_runs[rn]))
                    rn_std = np.std(res_runs[rn]) / np.mean(res_runs[rn]) * 100
                    feature_values[f'{rn}_std'].append(rn_std)

                flops = prof_item['flops'] / 1e6
                mem_bytes = prof_item['mem_bytes'] / 1024 / 1024
                feature_values['batch_size'].append(bs)
                feature_values['seq_len'].append(seq_len)
                feature_values['energy_mean'].append(energy_mean)
                feature_values['energy_std'].append(energy_std)
                feature_values['gpu_power_mean'].append(gpu_power_mean)
                feature_values['gpu_power_std'].append(gpu_power_std)
                feature_values['flops'].append(flops)
                feature_values['mem_bytes'].append(mem_bytes)
                feature_values['times_mean'].append(times_mean)
                feature_values['times_std'].append(times_std)
                feature_values['level_name'].append(prof_item['name'])
                feature_values['model_name'].append(model_name)

                print(f"b{bs},i{seq_len}:{prof_item['type']},"
                      f" {energy_std:.1f}%, {energy_mean / 10:.2f} J,"
                      f" {flops:.2f} MFlops,"
                      f" {mem_bytes:.2f} MiB,"
                      f" {prof_item['name']}")
    info = pd.DataFrame(data=feature_values)
    info.to_csv(feature_file)
    print(f'{model_name} done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
                        help="batch size")
    parser.add_argument("-bs", "--batch_step", type=int, default=2,
                        help="batch size step")
    parser.add_argument("-ss", "--seq_step", type=int, default=16,
                        help="input size step")
    parser.add_argument("-i", "--input_length", type=int, default=384,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="iterations to run the model")
    parser.add_argument("-m", "--model", type=str,
                        help="model name supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-l", "--level", type=str, default='linear',
                        help="level to use")
    main(parser.parse_args())
