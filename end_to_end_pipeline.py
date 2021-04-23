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
from common import get_hw_energy
from common import sanitize
import torch
import gc
import subprocess
import os
import bisect
import pickle
import pandas as pd
import numpy as np
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import get_module_info
from run_level_exp import run_ml_or_module
from visualise_model_as_graph import run_model_to_graph

res_names = ['cpu', 'mem', 'gpu', 'gpu_mem', 'gpu_clk', 'gpu_mem_clk']
feature_names = ['batch_size', 'seq_len', 'flops',
                'mem_bytes'] + res_names + \
                ['times_mean',
                'gpu_energy_mean',
                'level_name', 'level_type', 'model_name']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=384,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=3,
                        help="iterations to run the model")
    parser.add_argument("-n", "--probe_repeats", type=int, default=3,
                        help="initial probing iterations to run the model")
    parser.add_argument("-m", "--model_name", type=str,
                        help="model string supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-c", "--cache_dir", type=str,
                        help="Directory to cache model into")
    parser.add_argument("--res_file", type=str,
                        help="File to log resource usage into; "
                             "Used by rprof profiler")
    parser.add_argument("-nc", "--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    parser.add_argument("-le", "--log_energy_consumption", action="store_true",
                        help="Whether to track energy consumption")
    parser.add_argument("-mg", "--multi_gpu", action="store_true",
                        help="Whether to all gpus or not")
    args, _ = parser.parse_known_args()
    return args

def process_record(prof_info, res, feature_values,
                   model_name, bs, runs, seq_len):
    res_np = res.to_numpy()
    res_t = res_np[:, 0]
    for prof_item in prof_info:
        gpu_power_runs = []
        times_runs = []
        res_runs = {k: [] for k in res_names}
        repeats = prof_item['repeats']
        for r in range(1, runs + 1):
            start_r = prof_item[f'start_{r}']
            end_r = prof_item[f'end_{r}']
            times_runs.append((end_r - start_r) / repeats)

            res_s = bisect.bisect_right(res_t, start_r)
            res_e = bisect.bisect_right(res_t, end_r)
            res_r = res[res_s:res_e]
            for rn in res_names:
                res_runs[rn].append(res_r[rn].mean())
            gpu_power_r = res_r['gpu_power'].div(repeats * 1e3).sum() * 0.17
            gpu_power_runs.append(gpu_power_r)

        times_mean = np.mean(times_runs)
        gpu_power_mean = np.mean(gpu_power_runs)
        for rn in res_names:
            feature_values[rn].append(np.mean(res_runs[rn]))

        flops = prof_item['flops'] / 1e6
        mem_bytes = prof_item['mem_bytes'] / 1024 / 1024
        feature_values['batch_size'].append(bs)
        feature_values['seq_len'].append(seq_len)
        feature_values['gpu_energy_mean'].append(gpu_power_mean)
        feature_values['flops'].append(flops)
        feature_values['mem_bytes'].append(mem_bytes)
        feature_values['times_mean'].append(times_mean)
        feature_values['level_name'].append(prof_item['name'])
        feature_values['level_type'].append(prof_item['level_type'])
        feature_values['model_name'].append(model_name)

def end_to_end(model_name, batch_size, input_len, device, multi_gpu, runs, probe_repeats, res_file):

    root, tree, _ = run_model_to_graph(model_name, device)
    print(tree.keys())

    # run profiler
    print('Run profiler')
    prof_cmd = 'cd rprof; ./rprof 170 ' + res_file + ' 50 &'
    os.system(prof_cmd)

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
                gc.collect()
                torch.cuda.empty_cache()
                if prof_info is None:
                    continue
                prof_info['type'] = level_name
                prof_info['level_type'] = level_type
                model_prof_info.append(prof_info)

    # res_file - resources log file (res.csv)
    print('Read resources log file')
    res = pd.read_csv("rprof/"+res_file)
    res = res.apply(pd.to_numeric)

    feature_values = {k: [] for k in feature_names}
    process_record(model_prof_info, res, feature_values, model_name, batch_size, runs, input_len)

    # feature_values = json.load(open('temp.json', 'r'))

    # feature attributes from earlier stages of end to end pipeline
    # feature_values - dict_keys(['batch_size', 'seq_len', 'flops', 'mem_bytes', 
    # 'cpu', 'mem', 'gpu', 'gpu_mem', 'gpu_clk', 'gpu_mem_clk', 'times_mean', 
    # 'gpu_energy_mean', 'level_name', 'level_type', 'model_name'])
    print('Reconciling features with model graph now')
    num_feature_records = len(feature_values['level_name'])
    id_to_feature_map = {}
    for i in range(num_feature_records):
        name = feature_values['level_name'][i]
        scope, module_type = name.split(':')
        identifier = 'root.'+scope
        identifier = identifier.replace('layer.', '')

        # hack to make both naming conventions compatible
        if module_type == 'MatMul' or module_type == 'Softmax':
            identifier = identifier + '.' + module_type.lower()
        if identifier in id_to_feature_map:
            print("Hello "+name.replace('layer.', ''))
            print(identifier)

        feature_map = {}
        feature_map['flops'] = feature_values['flops'][i]
        feature_map['mem_bytes'] = feature_values['mem_bytes'][i]
        feature_map['cpu'] = feature_values['cpu'][i]
        feature_map['mem'] = feature_values['mem'][i]
        feature_map['gpu'] = feature_values['gpu'][i]
        feature_map['gpu_mem'] = feature_values['gpu_mem'][i]
        feature_map['gpu_clk'] = feature_values['gpu_clk'][i]
        feature_map['gpu_mem_clk'] = feature_values['gpu_mem_clk'][i]
        feature_map['times_mean'] = feature_values['times_mean'][i]
        feature_map['gpu_energy_mean'] = feature_values['gpu_energy_mean'][i]
        feature_map['level_type'] = feature_values['level_type'][i]
        id_to_feature_map[identifier] = feature_map

    # reconcile monitored features with tree dictionary
    print("Stuff in tree not in feature dictionary")
    for identifier, node in tree.items():
        try:
            features = id_to_feature_map[identifier]
            node.flops = features['flops']
            node.mem_bytes = features['mem_bytes']
            node.cpu = features['cpu']
            node.mem = features['mem']
            node.gpu = features['gpu']
            node.gpu_mem = features['gpu_mem']
            node.gpu_clk = features['gpu_clk']
            node.gpu_mem_clk = features['gpu_mem_clk']
            node.times_mean = features['times_mean']
            node.gpu_energy_mean = features['gpu_energy_mean']
            node.level_type = features['level_type']
        except:
            print(identifier)

    add_features_to_tree(root, id_to_feature_map)

    os.system("rm -rf rprof/"+res_file)

    return root, tree

def find_leaf_nodes(tree):
    
    """
    Find leaf nodes in model tree in left to right order of execution
    """

    leaf_nodes = []
    for identifier, node in tree.items():
        if len(node.child_nodes) == 0:
            leaf_nodes.append(node)

    return leaf_nodes

def add_features_to_tree(node, id_to_feature_map):

    """
    Add captured features to root object
    """

    if node:
        try:
            # not every node in root has corresponding entry in feature map
            identifier = node.scope
            features = id_to_feature_map[identifier]
            node.flops = features['flops']
            node.mem_bytes = features['mem_bytes']
            node.cpu = features['cpu']
            node.mem = features['mem']
            node.gpu = features['gpu']
            node.gpu_mem = features['gpu_mem']
            node.gpu_clk = features['gpu_clk']
            node.gpu_mem_clk = features['gpu_mem_clk']
            node.times_mean = features['times_mean']
            node.gpu_energy_mean = features['gpu_energy_mean']
            node.level_type = features['level_type']
        except:
            print("Stuff in root not in feature map")
            print(identifier)

        for child in node.child_nodes:
            add_features_to_tree(child, id_to_feature_map)
    
def main(args):
    start_time = time.time()
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
    res_file = args.res_file
    cache_dir = args.cache_dir
    root, tree = end_to_end(model_name, bs, seq_len, device, multi_gpu, runs, probe_repeats, res_file)
    if os.path.exists(os.path.join(cache_dir, model_name+"_"+str(seq_len)+"_"+str(bs))):
        end_time = time.time()
    else:
        root, tree = end_to_end(model_name, bs, seq_len, device, multi_gpu, runs, probe_repeats, res_file)
        leaf_nodes = find_leaf_nodes(tree)
        end_time = time.time()
        print(f'Takes {end_time-start_time} seconds to run')
        print('Saving to pickle files')
        pickle.dump([root, tree, leaf_nodes], open(os.path.join(cache_dir, model_name+"_"+str(seq_len)+"_"+str(bs)), "wb"))
    final_time = time.time()
    print(f'Takes {final_time-end_time} seconds to finish everything after logging step')

if __name__ == '__main__':
    args = parse_args()
    main(args)
