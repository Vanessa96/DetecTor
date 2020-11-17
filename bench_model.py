#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import time
import argparse
import json
from pathlib import Path
import torch
from transformers import BertModel
from transformers import AutoConfig

start_times = dict()
end_times = dict()
start_mem = dict()
end_mem = dict()


def log_start_builder(name, cu_mem):
    def log_start(m, _m_in):
        start_times[f'{name}:{m.__class__.__name__}'] = time.perf_counter()
        if cu_mem:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            mem_s = torch.cuda.memory_stats()
            start_mem[f'{name}:{m.__class__.__name__}'] = mem_s

    return log_start


def log_end_builder(name, cu_mem):
    def log_end(m, _m_in, _m_out):
        end_times[f'{name}:{m.__class__.__name__}'] = time.perf_counter()
        # print(name, m.__class__.__name__, 'end', time.perf_counter())
        if cu_mem:
            torch.cuda.empty_cache()
            mem_e = torch.cuda.memory_stats()
            end_mem[f'{name}:{m.__class__.__name__}'] = mem_e

    return log_end


def main(args):
    # model_name = '"prajjwal1/bert-tiny"'
    model_name = 'bert-base-uncased'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else args.n_gpu

    config = AutoConfig.from_pretrained(model_name)
    config.hidden_act = 'gelu_fast'
    config.torchscript = True
    model = BertModel(config)
    seq_len = args.input_length
    bs = args.batch_size
    input_ids = torch.randint(1000, size=(bs, seq_len), dtype=torch.long,
                              device=device)
    # token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long,
    #                              device=device)
    # pos_ids = torch.arange(config.max_position_embeddings,
    #                        device=device).expand((1, -1))[:, :seq_len]
    model = model.eval()
    model.to(device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    cu_mem = args.cuda_memory
    profile = args.profile
    if cu_mem:
        print('profiling cuda memory')

    if profile:
        runs = args.runs
        file_prefix = f'{model_name}-r{runs}-b{bs}-i{seq_len}'
        timings_file = out_dir.joinpath(f'{file_prefix}-timings.json')
        start_timings = dict()
        end_timings = dict()
        start_mem_info = dict()
        end_mem_info = dict()
        for _ in range(3):
            _ = model(input_ids)  # warmup
        for name, module in model.named_modules():
            # print(name, module.__class__.__name__)
            module.register_forward_pre_hook(log_start_builder(name, cu_mem))
            module.register_forward_hook(log_end_builder(name, cu_mem))
        for run in range(runs):
            _ = model(input_ids)
            for k, start in start_times.items():
                duration = (end_times[k] - start) * 1000
                start_timings[f'{run}-{k}'] = start
                end_timings[f'{run}-{k}'] = end_times[k]
                print(f'{run}-{k}, {duration:.3f} ms, {start}, {end_times[k]}')
                if cu_mem:
                    start_mem_info[f'{run}-{k}'] = start_mem[k]
                    end_mem_info[f'{run}-{k}'] = end_mem[k]
                    # print(f'{run}-{k}, {start_mem[k]}, {end_mem[k]}')
        timings = json.dumps({'start_timings': start_timings,
                              'end_timings': end_timings,
                              'start_mem_info': start_mem_info,
                              'end_mem_info': end_mem_info,
                              'keys': list(start_times.keys()),
                              'runs': args.runs})
        timings_file.write_text(timings)
    else:  # trace only to get the graph and statistics like flops, mem_static
        file_prefix = f'{model_name}-b{bs}-i{seq_len}'
        cg_file = out_dir.joinpath(f'{file_prefix}-cg.txt')
        trace = torch.jit.trace(model, input_ids)
        graph = trace.inlined_graph
        cg_file.write_text(str(graph))
        # todo: import cg/node, construct aggregated graph with flops features
        # cg_features_file = out_dir.joinpath(f'{model_name}-cg-feat.csv')


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
