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

import logging

logger = logging.getLogger('nrg')

logger.setLevel(logging.INFO)
fmt_str = "%(levelname)s:%(asctime)s.%(msecs)03d:%(pathname)s:%(lineno)d: " \
          "%(message)s"
fmt = logging.Formatter(fmt_str, "%Y-%m-%d_%H:%M:%S")
handler = logging.StreamHandler()
handler.setFormatter(fmt)
logger.addHandler(handler)


def get_flops_mem_bytes(graph):
    flops = dict()
    mem_bytes = dict()

    for node in graph.nodes:
        flops[node.id] = node.flops
        mem_bytes[node.id] = node.mem_bytes
        # if node.flops:
        #     print(node.op, node.flops, node.mem_bytes)
    return sum(flops.values()), sum(mem_bytes.values())


level_sigs = set()


def run_level(level, num_repeats, runs, device):
    fn = level['module']
    fname = level['name']
    fi_size, fi_dtype = level['inputs']
    # fo_size, fo_dtype = level['outputs']
    if fi_dtype in (torch.int32, torch.int64, torch.long):
        fi = torch.zeros(fi_size, dtype=fi_dtype, device=device)
    else:
        fi = torch.rand(fi_size, dtype=fi_dtype, device=device)
    # fo = torch.rand(fo_size, dtype=fo_dtype)
    trace = torch.jit.trace(fn, fi)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, fname)
    flops, mem_bytes = get_flops_mem_bytes(graph)
    level_type = fname.split(':')[-1]
    sig = f"{level_type},{flops},{mem_bytes}"
    level_prof = dict(name=fname, flops=flops, mem_bytes=mem_bytes)

    if flops > 0:
        # should in the range [100, 5e6]
        adjusted_repeats = max(num_repeats // (flops / 1e9), 100)
        calibrated_repeats = int(min(adjusted_repeats, 5e6))
    # elif mem_bytes > 0:
    #     mem_mb = mem_bytes / 1024 / 1024
    #     calibrated_repeats = int(num_repeats // (mem_mb / 100))
    else:
        calibrated_repeats = num_repeats

    if sig in level_sigs:
        logger.info(f'already profiled {sig} level, skip')
        return None
    else:
        level_sigs.add(sig)
    level_prof['repeats'] = calibrated_repeats
    logger.info(f'{fname} flops={flops} mem_bytes={mem_bytes}, '
                f'repeats={calibrated_repeats}')
    for run in range(1, runs + 1):
        logger.info(f'run ({run}/{runs}) {fname} levels')
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(calibrated_repeats):
            _ = fn(fi)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        level_prof[f'start_{run}'] = level_start
        level_prof[f'end_{run}'] = level_end
        time.sleep(1)  # sleep 1s to cool down
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
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and use_cuda else "cpu")
    seq_len = args.input_length
    bs = args.batch_size
    for model_name in args.models:
        logger.info(f'profiling {model_name} ml levels on {device}...')
        information = calibrate_e_ml(model_name, bs, seq_len, device)
        model_prof_info = []
        model_name = model_name.replace('/', '_')
        filename = f'{model_name}_level_r{runs}_b{bs}_i{seq_len}.json'
        prof_info_file = out_dir.joinpath(filename)
        for level_type, levels in information.items():
            # todo: support all ML levels
            if level_type != 'linear':
                continue
            for level in levels:
                prof_info = run_level(level, num_repeats, runs, device)
                if prof_info is None:
                    continue
                prof_info['type'] = level_type
                model_prof_info.append(prof_info)
        prof_info_file.write_text(json.dumps(model_prof_info))
        logger.info(f'{model_name} done.')
    logger.info('all done.')


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
    parser.add_argument("-n", "--num_repeats", type=int, default=10000,
                        help="iterations to run the model")
    parser.add_argument("-m", "--models", type=str, nargs='+',
                        help="list of model strings supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-nc", "--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    main(parser.parse_args())
