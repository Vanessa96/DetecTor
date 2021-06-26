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
import signal
import os
import bisect
import pickle
import pandas as pd
import numpy as np
import json
import copy
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import get_module_info
from run_level_exp import run_ml_or_module, run_model
from visualise_model_as_graph import run_model_to_graph

res_names = ['cpu', 'mem', 'gpu', 'gpu_mem', 'gpu_clk', 'gpu_mem_clk']
feature_names = ['batch_size', 'seq_len', 'flops',
                'mem_bytes'] + res_names + \
                ['times_mean',
                'gpu_energy_mean',
                'level_name', 'type', 'model_name']

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
    parser.add_argument("-j", "--json_dir", type=str,
                        help="Directory to save model as JSON into")
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
        feature_values['type'].append(prof_item['type'])
        feature_values['model_name'].append(model_name)

def flatten_node_to_json(obj):

    """
    Convert node to JSON object
    """

    data = {}
    data['scope'] = obj.scope
    data['level'] = obj.level
    data['parent_name'] = obj.parent_name
    data['child_nodes'] = []
    for child in obj.child_nodes:
        data['child_nodes'].append(child.scope)
    data['flops'] = obj.flops
    data['mem_bytes'] = obj.mem_bytes
    data['cpu'] = obj.cpu
    data['mem'] = obj.mem
    data['gpu'] = obj.gpu
    data['gpu_mem'] = obj.gpu_mem
    data['gpu_clk'] = obj.gpu_clk
    data['gpu_mem_clk'] = obj.gpu_mem_clk
    data['times_mean'] = obj.times_mean
    data['gpu_energy_mean'] = obj.gpu_energy_mean
    data['type'] = obj.type

    return data

def flatten_leaf_nodes(objs):
    
    """
    Convert list of leaf nodes to JSON object
    """

    data = {}
    for obj in objs:
        json_obj = flatten_node_to_json(obj)
        data[obj.scope] = json_obj
    return data

def flatten_tree(tree):

    """
    Convert tree to flat JSON object
    """

    data = {}
    for k,v in tree.items():
        json_obj = flatten_node_to_json(v)
        data[k] = json_obj
    return data

def end_to_end(model_name, batch_size, input_len, device, multi_gpu, runs, probe_repeats, res_file):

    """
    Run full end to end pipeline
    It converts model to graph, then extracts features into files, reconciles features and returns the root and tree objects
    """

    root, tree, _ = run_model_to_graph(model_name, device)

    # run profiler
    print('Run profiler')
    prof_cmd = "cd rprof/; ./rprof 170 "+res_file+" 10000"
    print(prof_cmd)
    proc = subprocess.Popen(prof_cmd, shell=True)
    print(f'Profiler process ID: {proc.pid}')

    model_prof_info = []
    # open a features file here and log everything after run_* function call
    level_types = ['ml', 'ml-np', 'module']
    for level_type in level_types:
        # call get_module_info with level_type=ml and ml-np
        print(f'Getting module information for {level_type}')
        information = get_module_info(model_name, batch_size, input_len, device, level_type)

        # call run_ml_or_module with level_type=ml and ml-np and module
        for level_name, levels in information.items():
            for level in levels:
                # possible duplication of module data due to calling ml and ml-np
                print(f'Running {level_name}')
                prof_info = run_ml_or_module(model_name, batch_size, input_len,
                                             probe_repeats, runs, device,
                                             level, level_name, multi_gpu)
                gc.collect()
                torch.cuda.empty_cache()
                if prof_info is None:
                    continue
                prof_info['type'] = level_type
                # import pdb; pdb.set_trace()
                model_prof_info.append(prof_info)
    print('Running model')
    prof_info = run_model(model_name, batch_size, input_len,
                              probe_repeats, runs, device, multi_gpu)
    prof_info['type'] = 'model'
    model_prof_info.append(prof_info)

    # res_file - resources log file (res.csv)
    print('Read resources log file')
    res = pd.read_csv("rprof/"+res_file)
    res = res.apply(pd.to_numeric)

    feature_values = {k: [] for k in feature_names}
    process_record(model_prof_info, res, feature_values, model_name, batch_size, runs, input_len)

    print('Reconciling features with model graph now')
    num_feature_records = len(feature_values['level_name'])
    id_to_feature_map = {}
    for i in range(num_feature_records):
        name = feature_values['level_name'][i]
        if name == model_name:
            identifier = 'root'
        else:
            scope, module_type = name.split(':')
            identifier = 'root.'+scope
        identifier = identifier.replace('layer.', '')

        # hack to make both naming conventions compatible
        if module_type == 'MatMul' or module_type == 'Softmax':
            identifier = identifier + '.' + module_type.lower()

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
        feature_map['type'] = feature_values['type'][i]
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
            node.type = features['type']
        except:
            print(identifier)

    add_features_to_tree(root, id_to_feature_map)

    return root, tree, proc

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
            node.type = features['type']
        except:
            print("Stuff in root not in feature map")
            print(identifier)

        for child in node.child_nodes:
            add_features_to_tree(child, id_to_feature_map)

def add_child_objects_into_node(node, tree_json):

    """
    For each child name, add the corresponding object to a new field in the node
    """

    node_children = []
    for child_node_name in node['child_nodes']:
        node_children.append(tree_json[child_node_name])
    node['child_nodes_obj'] = node_children

def add_empty_child_objects_into_node(node):

    """
    If a node does not have children, create an empty child_nodes_obj field that matches child_nodes field
    This can likely be integrated into add_child_objects_into_node() with an empty flag as well
    """

    node['child_nodes_obj'] = []

def create_frontend_compatible_data(root_json, tree_json):

    """
    Convert files to required nested JSON format supported by D3 frontend
    """

    if root_json['child_nodes'] == []:
        add_empty_child_objects_into_node(root_json)
        return
    add_child_objects_into_node(root_json, tree_json)
    for child_node_name in root_json['child_nodes']:
        create_frontend_compatible_data(tree_json[child_node_name], tree_json)

def save_files(json_dir, model_fname, tree_json, root_json, leaf_json):

    """
    Save requisite root, tree and leaf nodes object
    """

    with open(os.path.join(json_dir, model_fname+'_root.json'), 'w+') as fp:
        json.dump(root_json, fp)

    with open(os.path.join(json_dir, model_fname+'_leaf.json'), 'w+') as fp:
        json.dump(leaf_json, fp)

    with open(os.path.join(json_dir, model_fname+'_tree.json'), 'w+') as fp:
        json.dump(tree_json, fp)

def serve(model_name, seq_len, bs, json_dir, device, multi_gpu, runs, probe_repeats, res_file, start_time):

    """
    Function called by API
    """

    model_fname = model_name.replace("/", "-")
    fname = model_fname+"_"+str(seq_len)+"_"+str(bs)
    if os.path.exists(os.path.join(json_dir, fname+"_root.json")) and os.path.exists(os.path.join(json_dir, fname+"_leaf.json")) and os.path.exists(os.path.join(json_dir, fname+"_tree.json")):
        end_time = time.time()
        print(f'Takes {end_time-start_time} seconds to run')
        with open(os.path.join(json_dir, fname+'_root.json'), 'r') as fp:
            root_json = json.load(fp)

        with open(os.path.join(json_dir, fname+'_leaf.json'), 'r') as fp:
            leaf_json = json.load(fp)

        with open(os.path.join(json_dir, fname+'_tree.json'), 'r') as fp:
            tree_json = json.load(fp)

    else:
        root, tree, proc = end_to_end(model_name, bs, seq_len, device, multi_gpu, runs, probe_repeats, res_file)
        leaf_nodes = find_leaf_nodes(tree)
        end_time = time.time()
        print(f'Takes {end_time-start_time} seconds to run')

        # save root, tree and leaf nodes into pickle objects for future use

        root_json = flatten_node_to_json(root)
        leaf_json = flatten_leaf_nodes(leaf_nodes)
        tree_json = flatten_tree(tree)

        save_files(json_dir, fname, tree_json, root_json, leaf_json)

    compatible_tree = copy.deepcopy(root_json)

    create_frontend_compatible_data(compatible_tree, tree_json)

    with open(os.path.join(json_dir, fname+'_frontend_tree.json'), 'w+') as fp:
        json.dump(compatible_tree, fp)

    final_time = time.time()

    try:
        # these variables don't exist if the files are loaded from memory
        kill_pid = proc.pid+1
        print(f'Killing process {kill_pid}')
        os.kill(kill_pid, signal.SIGTERM)
        del_cmd = "rm rprof/"+res_file
        print(del_cmd)
        os.system(del_cmd)
        # sleep needed to allow process kill before next command is run (wrt batch shell script)
        time.sleep(3)
    except:
        pass

    print(f'Takes {final_time-end_time} seconds to finish everything after logging step')

    return root_json, tree_json, leaf_json, compatible_tree
    
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
    json_dir = args.json_dir

    root_json, tree_json, leaf_json, compatible_tree = serve(model_name, seq_len, bs, json_dir, device, multi_gpu, runs, probe_repeats, res_file, start_time)

if __name__ == '__main__':
    args = parse_args()
    main(args)
