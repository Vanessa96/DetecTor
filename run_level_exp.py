#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import calibrate_e_ml
from cg.node import construct_aggregation_graph
from common import logger
from common import sanitize


def get_flops_mem_bytes(graph):
    flops = dict()
    mem_bytes = dict()

    for node in graph.nodes:
        flops[node.id] = node.flops
        mem_bytes[node.id] = node.mem_bytes
        # if node.flops:
        #     print(node.op, node.flops, node.mem_bytes)
    return sum(flops.values()), sum(mem_bytes.values())


def get_model_flops_mem_bytes(module_fn, inputs, module_name):
    trace = torch.jit.trace(module_fn, inputs)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, module_name)
    flops, mem_bytes = get_flops_mem_bytes(graph)
    return flops, mem_bytes


def calibrate_repeats(flops, repeats):
    if flops > 0:
        # should in the range [100, 1e5]
        # todo: better heuristics!
        adjusted_repeats = max(repeats // (flops / 1e9), 100)
        calibrated_repeats = int(min(adjusted_repeats, 1e5))
    # elif mem_bytes > 0:
    #     mem_mb = mem_bytes / 1024 / 1024
    #     calibrated_repeats = int(num_repeats // (mem_mb / 100))
    else:
        calibrated_repeats = repeats
    return calibrated_repeats


def run_model(model_name, bs, seq_len, num_repeats, runs, device):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    model = model.eval().to(device)

    input_ids = torch.randint(1000, size=(bs, seq_len), dtype=torch.long,
                              device=device)
    inputs = (input_ids,)
    if config.model_type == 't5':
        #  attention_mask=None, decoder_input_ids=None
        inputs += (input_ids, input_ids)
    flops, mem_bytes = get_model_flops_mem_bytes(model, inputs, model_name)
    model_prof = dict(name=model_name, flops=flops, mem_bytes=mem_bytes)
    model_prof['repeats'] = num_repeats
    logger.info(f'{model_name}_b{bs}_i{seq_len}, '
                f'flops={flops}, mem_bytes={mem_bytes}, '
                f'repeats={num_repeats}')
    seq2seq = hasattr(model, 'decoder')
    kwargs = {'decoder_input_ids': input_ids} if seq2seq else {}
    for run in range(1, runs + 1):
        logger.info(f'run {model_name}_b{bs}_i{seq_len} ({run}/{runs})')
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(num_repeats):
            _ = model(input_ids, **kwargs)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        model_prof[f'start_{run}'] = level_start
        model_prof[f'end_{run}'] = level_end
        time.sleep(3)  # sleep 3s to cool down
    return model_prof


level_sigs = set()


def run_level(model_name, bs, seq_len, num_repeats, runs, device,
              level, level_type):
    # # uncomment support specific ML levels
    # if level_type != 'linear':
    #     return None
    fn = level['module']
    fname = level['name']
    fi_size, fi_dtype = level['inputs']
    # fo_size, fo_dtype = level['outputs']
    if fi_dtype in (torch.int32, torch.int64, torch.long):
        fi = torch.zeros(fi_size, dtype=fi_dtype, device=device)
    else:
        fi = torch.rand(fi_size, dtype=fi_dtype, device=device)
    # fo = torch.rand(fo_size, dtype=fo_dtype)
    flops, mem_bytes = get_model_flops_mem_bytes(fn, fi, fname)
    sig = f"{level_type},{flops},{mem_bytes}"
    level_prof = dict(name=fname, flops=flops, mem_bytes=mem_bytes)

    calibrated_repeats = calibrate_repeats(flops, num_repeats)

    if sig in level_sigs:
        logger.info(f'already profiled {sig} level, skip')
        return None
    else:
        level_sigs.add(sig)
    level_prof['repeats'] = calibrated_repeats
    logger.info(f'{model_name}_b{bs}_i{seq_len}_{level_type}, '
                f'flops={flops}, mem_bytes={mem_bytes}, '
                f'repeats={calibrated_repeats}, {fname}')
    for run in range(1, runs + 1):
        logger.info(f'run {model_name}_b{bs}_i{seq_len}_{level_type}, '
                    f'({run}/{runs}) {fname} levels')
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(calibrated_repeats):
            _ = fn(fi)
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        level_prof[f'start_{run}'] = level_start
        level_prof[f'end_{run}'] = level_end
        time.sleep(1)  # sleep 1s to cool down
    return level_prof


def run_module(model_name, bs, seq_len, num_repeats, runs, device):
    # todo module level exp
    module_prof = dict()
    return module_prof


def main(args):
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
    exp_type = args.exp_type
    for model_name in args.models:
        logger.info(f'profiling {model_name} ml levels on {device}...')
        information = calibrate_e_ml(model_name, bs, seq_len, device)
        model_prof_info = []
        model_name = sanitize(model_name)
        filename = f'{model_name}_{exp_type}_r{runs}_b{bs}_i{seq_len}.json'
        prof_info_file = out_dir.joinpath(filename)
        if exp_type == 'level':
            for level_type, levels in information.items():
                for level in levels:
                    prof_info = run_level(model_name, bs, seq_len,
                                          num_repeats, runs, device,
                                          level, level_type)
                    if prof_info is None:
                        continue
                    prof_info['type'] = level_type
                    model_prof_info.append(prof_info)
        elif exp_type == 'model':
            prof_info = run_model(model_name, bs, seq_len,
                                  num_repeats, runs, device)
            model_prof_info.append(prof_info)
        else:
            prof_info = run_module(model_name, bs, seq_len,
                                   num_repeats, runs, device)
            model_prof_info.append(prof_info)
        prof_info_file.write_text(json.dumps(model_prof_info))
        logger.info(f'{model_name} done.')
    logger.info('all done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-t", "--exp_type", type=str, required=True,
                        choices=('ml', 'module', 'model'),
                        help="ml, module, model type")
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
