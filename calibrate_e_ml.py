#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"

"""
The 'information' variable in the script contains all the requisite information for one run of the model
Example usage: python calibrate_e_ml.py --model-name bert-base-uncased --batch-size 2 --input-len 100 --no-cuda
Arguments:
    model-name: model name in huggingface repository (e.g. prajjwal1/bert-tiny)
    batch-size: batch size of input tensor, first parameter of shape for model input
    input-len: input length of input tensor, second parameter of shape for model input
    no-cuda: use if you don't want the script to use CUDA
"""

import os
import argparse
import csv
import time
import torch
from transformers import AutoModel
from transformers import AutoConfig
from collections import defaultdict

start_times = dict()
end_times = dict()
module_inputs = dict()
module_outputs = dict()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='Huggingface model name to work with (e.g. bert)', default='bert-base-uncased')
    parser.add_argument('--batch-size', type=int, help='Batch size of input to run model with', default=1)
    parser.add_argument('--input-len', type=int, help='Length of input to run model with', default=100)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true', help='Remove use of CUDA')
    args, _ = parser.parse_known_args()
    return args

def log_start_builder(name):
    def log_start(module, m_in):
        start_times[f'{name}:{module.__class__.__name__}'] = time.perf_counter()
    return log_start

def log_end_builder(name):
    def log_end(module, m_in, m_out):
        end_times[f'{name}:{module.__class__.__name__}'] = time.perf_counter()
        module_inputs[f'{name}:{module.__class__.__name__}'] = m_in[0].shape
        module_outputs[f'{name}:{module.__class__.__name__}'] = m_out.shape
    return log_end

def is_ml_operation(module):

    """
    This function checks if any given module is of a type that we want to analyse for E_ML operations
    """

    if isinstance(module, torch.nn.Linear):
        return True
    elif isinstance(module, torch.nn.LayerNorm):
        return True
    elif isinstance(module, torch.nn.Embedding):
        return True
    return False

def calibrate_e_ml(model_name, batch_size, input_len, cuda_available):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)

    torch.set_grad_enabled(False)

    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not cuda_available else "cpu")
    model = model.eval().to(device)

    inputs = torch.randint(1000, size=(batch_size, input_len)).long()

    for (name, module) in model.named_modules():
        if is_ml_operation(module):
            module.register_forward_pre_hook(log_start_builder(name))
            module.register_forward_hook(log_end_builder(name))

    loss = model(input_ids=inputs, return_dict=True)

    information = defaultdict(list)
    for module_name in start_times.keys():
        module_info = {}
        module_info['name'] = module_name
        module_info['inputs'] = module_inputs[module_name]
        module_info['outputs'] = module_outputs[module_name]
        module_info['runtime'] = end_times[module_name] - start_times[module_name]

        module_identifier = module_name.split(':')[-1]
        if module_identifier == 'Linear':
            information['linear'].append(module_info)
        elif module_identifier == 'LayerNorm':
            information['layernorm'].append(module_info)
        elif module_identifier == 'Embedding':
            information['embedding'].append(module_info)

    return information

def main(args):
    information = calibrate_e_ml(args.model_name, args.batch_size, args.input_len, args.no_cuda)

if __name__ == '__main__':
    args = parse_args()
    main(args)
