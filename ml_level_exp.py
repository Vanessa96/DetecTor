#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import json
import time
from pathlib import Path

import torch

from calibrate_e_ml import calibrate_e_ml
from cg.node import construct_aggregation_graph


def get_flops_mem_bytes(graph):
    flops = dict()
    mem_bytes = dict()

    for node in graph.nodes:
        flops[node.id] = node.flops
        mem_bytes[node.id] = node.mem_bytes
        if node.flops:
            print(node.op, node.flops, node.mem_bytes)
    return sum(flops.values()), sum(mem_bytes.values())


def run_level(level, num_repeats, runs):
    fn = level['module']
    fname = level['name']
    fi_size, fi_dtype = level['inputs']
    # fo_size, fo_dtype = level['outputs']
    fi = torch.rand(fi_size, dtype=fi_dtype)
    # fo = torch.rand(fo_size, dtype=fo_dtype)
    trace = torch.jit.trace(fn, fi)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, fname)
    flops, mem_bytes = get_flops_mem_bytes(graph)
    level_prof = dict(name=fname, flops=flops, mem_bytes=mem_bytes)
    for run in range(1, runs + 1):
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(num_repeats):
            _ = fn(fi)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        level_prof[f'start_{run}'] = level_start
        level_prof[f'end_{run}'] = level_end
        time.sleep(5)  # sleep 5s to cool down
    return level_prof


def main(args):
    # model_name = "prajjwal1/bert-tiny"
    # model_name = 'bert-base-uncased'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs
    num_repeats = args.num_repeats

    torch.set_grad_enabled(False)
    use_cuda = not args.no_cuda
    seq_len = args.input_length
    bs = args.batch_size
    for model_name in args.models:
        print(f'profiling {model_name} ml levels...')
        filename = f'{model_name}_level_r{runs}_b{bs}_i{seq_len}.json'
        prof_info_file = out_dir.joinpath(filename)
        information = calibrate_e_ml(model_name, bs, seq_len, use_cuda)
        model_prof_info = []
        for level_type, levels in information.items():
            for level in levels:
                prof_info = run_level(level, num_repeats, runs)
                prof_info['type'] = level_type
                model_prof_info.append(prof_info)
        prof_info_file.write_text(json.dumps(model_prof_info))
        print(f'{model_name} done.')
    print('all done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=384,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="iterations to run the model")
    parser.add_argument("-n", "--num_repeats", type=int, default=100000,
                        help="iterations to run the model")
    parser.add_argument("-m", "--models", type=str, nargs='+',
                        help="list of model strings supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-nc", "--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    main(parser.parse_args())
