#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import argparse
import json
import time
from functools import partial
from functools import update_wrapper
from pathlib import Path
from common import logger
from common import sanitize
import torch
from transformers import AutoConfig
from transformers import AutoModel

from calibrate_e_ml import get_module_info
from cg.node import construct_aggregation_graph


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


def calibrate_repeats(flops, level_name, repeats):
    if flops > 0:
        # should in the range [100, 1e6]
        # todo: better heuristics!
        adjusted_repeats = max(repeats // (flops / 1e9), 100)
        calibrated_repeats = int(min(adjusted_repeats, 1e6))
    # elif mem_bytes > 0:
    #     mem_mb = mem_bytes / 1024 / 1024
    #     calibrated_repeats = int(num_repeats // (mem_mb / 100))
    else:
        calibrated_repeats = repeats
    if level_name == 'embedding':
        calibrated_repeats *= 10
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


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def run_ml_or_module(model_name, bs, seq_len, num_repeats, runs,
                     level, level_name):
    # # uncomment support specific ML levels
    # if level_name != 'linear':
    #     return None
    fn = level['module']
    fname = level['name']
    fi = level['inputs']

    # separate tensor args and rest from fi
    # fixme: for now, tensor args are ordered first,
    #  if not, need another solution,  (not true for roberta)
    if isinstance(fi, dict):
        ti = [t for t in fi.values() if isinstance(t, torch.Tensor)]
        ri = {rk: r for rk, r in fi.items() if not isinstance(r, torch.Tensor)}
        # https://github.com/pytorch/pytorch/issues/14455#issuecomment-445962680
        fn.forward = wrapped_partial(fn.forward, **ri)
    elif isinstance(fi, tuple):
        ti = [t for t in fi if isinstance(t, torch.Tensor)]
        ri = [r for r in fi if not isinstance(r, torch.Tensor)]
        fn.forward = wrapped_partial(fn.forward, *ri)
    else:
        # unknown type
        ti = fi
        logger.warning(f'unknown fi: {type(fi)}')
    flops, mem_bytes = get_model_flops_mem_bytes(fn, ti, fname)
    sig = f"{level_name},{flops},{mem_bytes}"
    level_prof = dict(name=fname, flops=flops, mem_bytes=mem_bytes)

    calibrated_repeats = calibrate_repeats(flops, level_name, num_repeats)

    if sig in level_sigs:
        logger.info(f'already profiled {sig} level, skip')
        return None
    else:
        level_sigs.add(sig)
    level_prof['repeats'] = calibrated_repeats
    logger.info(f'{model_name}_b{bs}_i{seq_len}_{level_name}, '
                f'flops={flops}, mem_bytes={mem_bytes}, '
                f'repeats={calibrated_repeats}, {fname}')
    for run in range(1, runs + 1):
        logger.info(f'run {model_name}_b{bs}_i{seq_len}_{level_name}, '
                    f'({run}/{runs}) {fname} levels')
        level_start = time.clock_gettime(time.CLOCK_REALTIME)
        for _ in range(calibrated_repeats):
            if isinstance(fi, tuple):
                _ = fn(*fi)
            elif isinstance(fi, dict):
                _ = fn(**fi)
            else:
                logger.warning(f'{type(fi)}')
        level_end = time.clock_gettime(time.CLOCK_REALTIME)
        level_prof[f'start_{run}'] = level_start
        level_prof[f'end_{run}'] = level_end
        time.sleep(1)  # sleep 1s to cool down
    return level_prof


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
    level_type = args.level_type
    for model_name in args.models:
        logger.info(f'profiling {model_name} {level_type} on {device}...')
        information = get_module_info(model_name, bs, seq_len, device,
                                      level_type)
        model_prof_info = []
        model_name_s = sanitize(model_name)
        filename = f'{model_name_s}_{level_type}_r{runs}_b{bs}_i{seq_len}.json'
        prof_info_file = out_dir.joinpath(filename)
        if level_type == 'model':
            prof_info = run_model(model_name, bs, seq_len,
                                  num_repeats, runs, device)
            model_prof_info.append(prof_info)
        else:
            for level_name, levels in information.items():
                for level in levels:
                    prof_info = run_ml_or_module(model_name, bs, seq_len,
                                                 num_repeats, runs,
                                                 level, level_name)
                    if prof_info is None:
                        continue
                    prof_info['type'] = level_name
                    model_prof_info.append(prof_info)
        prof_info_file.write_text(json.dumps(model_prof_info))
        logger.info(f'{model_name} done.')
    logger.info('all done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-t", "--level_type", type=str, required=True,
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
