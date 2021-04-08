#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"


import argparse
import copy
import inspect
import json
import time
from functools import partial
from functools import update_wrapper
from pathlib import Path
from common import logger
from common import sanitize
import torch
import subprocess
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import get_module_info
from run_level_exp import run_ml_or_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=384,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="iterations to run the model")
    parser.add_argument("-n", "--probe_repeats", type=int, default=10,
                        help="initial probing iterations to run the model")
    parser.add_argument("-m", "--model_name", type=str,
                        help="model string supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-nc", "--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("-le", "--log_energy_consumption", action="store_true",
                        help="Whether to track energy consumption")
    parser.add_argument("-mg", "--multi_gpu", action="store_true",
                        help="Whether to all gpus or not")
    args, _ = parser.parse_known_args()
    return args

def end_to_end(model_name, batch_size, input_len, device, multi_gpu, runs, probe_repeats):

    # run profiler
    subprocess.call(['cd', 'rprof', './rprof', '170', 'res.csv', '50'])

    model_prof_info = []
    level_types = ['ml', 'ml-np']
    for level_type in level_types:
        # call get_module_info with level_type=ml and ml-np
        information = get_module_info(model_name, batch_size, input_len, device, level_type)

        # call run_ml_or_module with level_type=ml and ml-np and module
        for level_name, levels in information.items():
            for level in levels:
                prof_info = run_ml_or_module(model_name, batch_size, input_len,
                                             probe_repeats, runs, device,
                                             level, level_name, multi_gpu)
                if prof_info is None:
                    continue
                prof_info['type'] = level_name
                model_prof_info.append(prof_info)

        # merge these 3 dictionaries and create model tree
    import pdb; pdb.set_trace()

def main(args):
    torch.set_grad_enabled(False)
    use_cuda = not args.no_cuda
    multi_gpu = args.multi_gpu
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and use_cuda else "cpu")
    seq_len = args.input_length
    bs = args.batch_size
    model_name = args.model_name
    runs = args.runs
    probe_repeats = args.probe_repeats
    end_to_end(model_name, bs, seq_len, device, multi_gpu, runs, probe_repeats)

if __name__ == '__main__':
    args = parse_args()
    main(args)