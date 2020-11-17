#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import time
import argparse
from pathlib import Path
import torch
from transformers import BertModel
from transformers import AutoConfig

start_times = dict()
end_times = dict()


def log_start_builder(name):
    def log_start(m, _m_in):
        start_times[f'{name}:{m.__class__.__name__}'] = time.perf_counter()

    return log_start


def log_end_builder(name):
    def log_end(m, _m_in, _m_out):
        end_times[f'{name}:{m.__class__.__name__}'] = time.perf_counter()
        # print(name, m.__class__.__name__, 'end', time.perf_counter())

    return log_end


def main(args):
    # model_name = '"prajjwal1/bert-tiny"'
    model_name = 'bert-base-uncased'
    out_dir = Path(args.out_dir)
    cg_file = out_dir.joinpath(f'{model_name}-cg.txt')
    cg_features_file = out_dir.joinpath(f'{model_name}-cg-feat.txt')
    timings_file = out_dir.joinpath(f'{model_name}-timings.json')

    config = AutoConfig.from_pretrained(model_name)
    config.hidden_act = 'gelu_fast'
    config.torchscript = True
    model = BertModel(config)
    inputs = torch.randint(1000, size=(1, 100)).long()
    model = model.eval()

    for name, module in model.named_modules():
        print(name, module.__class__.__name__)
        module.register_forward_pre_hook(log_start_builder(name))
        module.register_forward_hook(log_end_builder(name))

    trace = torch.jit.trace(model, inputs)

    for k, start in start_times.items():
        duration = (end_times[k] - start) * 1000
        print(f'{k}, {duration:.3f} ms, {start}, {end_times[k]}')

    graph = trace.inlined_graph
    cg_file.write_text(graph)


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
    parser.add_argument("-p", "--profile", action='store_false',
                        help="profile the model runtime timings, "
                             "default to false;")
    parser.add_argument("-m", "--models", type=str, nargs='+',
                        help="list of model strings supported by the "
                             "HuggingFace Transformers library")
    parser.add_argument("-d", "--device", type=str, choices=("cpu", "gpu"),
                        help="output dir")
    main(parser.parse_args())
