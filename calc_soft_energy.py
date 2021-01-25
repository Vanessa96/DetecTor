#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import bisect
import json
from pathlib import Path

from experiment_impact_tracker.utils import gather_additional_info
from experiment_impact_tracker.utils import load_initial_info
import numpy as np

from common import get_hw_energy
from common import sanitize


def one_energy(log_dir):
    sys_data = load_initial_info(log_dir)
    eff_data = gather_additional_info(sys_data, log_dir)
    return eff_data['total_power']


def main(args):
    energy_output_dir = Path(args.energy_output_dir)
    sw_energy = one_energy(energy_output_dir)

    runs = args.runs
    seq_len = args.input_length
    bs = args.batch_size
    model_name = args.model_name
    # print(f'{model_name} energy (J): {energy * 3.6e6:.2f}')
    energy_file = args.energy_file
    hw_energy = get_hw_energy(energy_file)

    model_name_s = sanitize(model_name)
    filename = f'{model_name_s}_model_r{runs}_b{bs}_i{seq_len}.json'
    profile_dir = Path(args.profile_dir)
    prof_info_file = profile_dir.joinpath(filename)
    with open(prof_info_file) as f:
        prof_item = json.load(f)[0]
        repeats = prof_item['repeats']
    sw_avg_energy = sw_energy * 3.6e6 / repeats

    energy_np = hw_energy.to_numpy()
    energy_t = energy_np[:, 0]
    energy_runs = []
    for r in range(1, runs + 1):
        start_r = prof_item[f'start_{r}']
        end_r = prof_item[f'end_{r}']
        e_s = bisect.bisect_right(energy_t, start_r)
        e_e = bisect.bisect_right(energy_t, end_r)
        # sampling rate is 170 ms
        energy_r = hw_energy[e_s:e_e]['value'].div(repeats).sum() * 0.17
        energy_runs.append(energy_r)
    energy_mean = np.mean(energy_runs)
    print(f'sw_energy: {sw_avg_energy:.2f}, hw_energy: {energy_mean:.2f} (J)'
          f' for {model_name_s}_r{runs}_b{bs}_i{seq_len} avg over {repeats}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--energy_output_dir", type=str,
                        help="log dir from experiment_imapct_tracker")
    parser.add_argument("-p", "--profile_dir", type=str, )
    parser.add_argument("-ef", "--energy_file", type=str, )
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=32,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=1,
                        help="iterations to run the model")
    parser.add_argument("-m", "--model_name", type=str,
                        help="model string supported by the "
                             "HuggingFace Transformers library")
    main(parser.parse_args())
