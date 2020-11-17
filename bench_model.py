#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import csv
import time
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModel
from transformers import AutoConfig

from cg.node import construct_graph


def log_builder(name, timings, global_repeats, pre_hook,
                cu_mem=False, mem_stats=None):
    # handle shared module compute,
    # use global_repeats to track module call times, tested for albert

    def log(m, _m_in):
        module_key = f'{name}:{m.__class__.__name__}'

        repeats = global_repeats.get(module_key, -1)
        repeats += 1
        global_repeats[module_key] = repeats
        log_key = f'{module_key}:{repeats}'
        timings[log_key] = time.perf_counter()
        if cu_mem:
            # torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats()
            mem_s = torch.cuda.memory_stats()
            mem_stats[log_key] = mem_s

    def post_hook(m, _m_in, _m_out):
        return log(m, _m_in)

    return log if pre_hook else post_hook


def profile_model(model, input_ids, runs, cu_mem):
    start_timings = dict()
    end_timings = dict()
    start_mem_info = dict()
    end_mem_info = dict()
    for _ in range(3):
        _ = model(input_ids)  # warmup
    if cu_mem:
        print('profiling cuda memory')

    model_start_timings = dict()
    model_end_timings = dict()
    model_start_mem_stats = dict()
    model_end_mem_stats = dict()
    global_pre_repeats = dict()
    global_post_repeats = dict()

    for name, module in model.named_modules():
        # print(name, module.__class__.__name__)
        start_logger = log_builder(name, model_start_timings,
                                   global_pre_repeats, True,
                                   cu_mem, model_start_mem_stats)
        module.register_forward_pre_hook(start_logger)
        end_logger = log_builder(name, model_end_timings,
                                 global_post_repeats, False,
                                 cu_mem, model_end_mem_stats)
        module.register_forward_hook(end_logger)
    for run in range(runs):
        model_start_timings.clear()
        model_end_timings.clear()
        model_start_mem_stats.clear()
        model_end_mem_stats.clear()
        global_pre_repeats.clear()
        global_post_repeats.clear()
        _ = model(input_ids)
        for k, start in model_start_timings.items():
            duration = (model_end_timings[k] - start) * 1000
            start_timings[f'{run}-{k}'] = start
            end_timings[f'{run}-{k}'] = model_end_timings[k]
            print(f'{run}-{k}, {duration:.3f} ms, '
                  f'{start}, {model_end_timings[k]}')
            if cu_mem:
                start_mem_info[f'{run}-{k}'] = model_start_mem_stats[k]
                end_mem_info[f'{run}-{k}'] = model_end_mem_stats[k]
                # print(f'{run}-{k}, {start_mem[k]}, {end_mem[k]}')
    prof_info = json.dumps({'start_timings': start_timings,
                            'end_timings': end_timings,
                            'start_mem_info': start_mem_info,
                            'end_mem_info': end_mem_info,
                            'keys': list(model_start_timings.keys()),
                            'runs': runs})
    return prof_info


def analyze_model(trace_graph, model_name):
    graph_features = dict()
    # todo: import cg/node, construct graph with flops features
    graph, ops = construct_graph(trace_graph, model_name)
    for n in graph.nodes:
        # todo: design feature format
        graph_features[n.scope] = n.op
    return graph_features


def write_graph_features(features, output_file):
    with open(output_file, mode='w') as f:
        keys = ['name', 'age', 'job', 'city']  # fixme: use real feature names
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()  # add column names in the CSV file
        for feat in features:
            # todo: design feat format, list of keys
            writer.writerow(feat)


def main(args):
    # model_name = '"prajjwal1/bert-tiny"'
    # model_name = 'bert-base-uncased'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.set_grad_enabled(False)
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else args.n_gpu
    seq_len = args.input_length
    bs = args.batch_size
    input_ids = torch.randint(1000, size=(bs, seq_len), dtype=torch.long,
                              device=device)
    # token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long,
    #                              device=device)
    # pos_ids = torch.arange(config.max_position_embeddings,
    #                        device=device).expand((1, -1))[:, :seq_len]
    for model_name in args.models:
        print(f'benchmarking {model_name}...')
        config = AutoConfig.from_pretrained(model_name)
        config.torchscript = True
        model = AutoModel.from_config(config)
        model = model.eval().to(device)
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        cu_mem = args.cuda_memory
        profile = args.profile
        if profile:
            runs = args.runs
            file_prefix = f'{model_name}_r{runs}_b{bs}_i{seq_len}'
            prof_info_file = out_dir.joinpath(f'{file_prefix}_timings.json')
            prof_info = profile_model(model, input_ids, runs, cu_mem)
            prof_info_file.write_text(prof_info)
        else:  # jit trace to get the graph statistics like flops, mem_bytes
            file_prefix = f'{model_name}_b{bs}_i{seq_len}'
            cg_file = out_dir.joinpath(f'{file_prefix}_cg.txt')
            trace = torch.jit.trace(model, input_ids)
            graph = trace.inlined_graph
            cg_file.write_text(str(graph))
            graph_features = analyze_model(graph, model_name)
            cg_features_file = out_dir.joinpath(f'{model_name}_features.csv')
            write_graph_features(graph_features, cg_features_file)
        print(f'{model_name} done.')
    print('all done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="output dir")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                        help="batch size")
    parser.add_argument("-i", "--input_length", type=int, default=128,
                        help="input sequence length")
    parser.add_argument("-r", "--runs", type=int, default=10,
                        help="iterations to run the model")
    parser.add_argument("-p", "--profile", action='store_true',
                        help="profile the model runtime timings, "
                             "default to false, trace only;")
    parser.add_argument("-cm", "--cuda_memory", action='store_true',
                        help="profile the runtime cuda memory, "
                             "default to false;")
    parser.add_argument("-m", "--models", type=str, nargs='+',
                        help="list of model strings supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-ng", "--n_gpu", type=int, default=1,
                        help="output dir")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Whether not to use CUDA when available")
    main(parser.parse_args())
