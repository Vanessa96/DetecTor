#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"

"""
The 'information' variable in the script contains all the requisite information
 for one run of the model
Example usage: 
python calibrate_e_ml.py --model-name bert-base-uncased --batch-size 2 --input-len 100 --no-cuda
Arguments:
    model-name: model name in huggingface repository (e.g. prajjwal1/bert-tiny)
    batch-size: batch size of input tensor, first parameter of shape for model input
    input-len: input length of input tensor, second parameter of shape for model input
    no-cuda: use if you don't want the script to use CUDA
"""

import argparse
import time
from collections import defaultdict

import torch
from torch import nn
from transformers import AutoConfig
from transformers import AutoModel

start_times = dict()
end_times = dict()
module_inputs = dict()
module_outputs = dict()
modules = dict()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str,
                        help='Huggingface model name to work with '
                             '(e.g. bert-base-uncased, distilbert-base-uncased,'
                             'google/mobilebert-uncased, roberta-base,'
                             'bert-large-uncased, roberta-large,'
                             'xlnet-base-cased, xlnet-large-cased,'
                             'albert-base-v2, albert-large-v2, t5-small, t5-base,'
                             'openai-gpt, gpt2, sshleifer/tiny-gpt2, distilgpt2'
                             'sshleifer/tiny-ctrl, facebook/bart-base, facebook/bart-large,'
                             'sshleifer/distilbart-xsum-6-6, valhalla/distilbart-mnli-12-3',
                        default='bert-base-uncased')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size of input to run model with', default=1)
    parser.add_argument('--input-len', type=int,
                        help='Length of input to run model with', default=100)
    parser.add_argument('--no-cuda', dest='no_cuda', action='store_true',
                        help='Remove use of CUDA')
    args, _ = parser.parse_known_args()
    return args


def log_start_builder(name):
    def log_start(module, m_in):
        start_times[f'{name}:{module.__class__.__name__}'] = time.perf_counter()

    return log_start


def log_end_builder(name):
    def log_end(module, m_in, m_out):
        end_times[f'{name}:{module.__class__.__name__}'] = time.perf_counter()
        module_inputs[f'{name}:{module.__class__.__name__}'] = (
            m_in[0].shape, m_in[0].dtype)
        module_outputs[f'{name}:{module.__class__.__name__}'] = (
            m_out.shape, m_out.dtype)
        modules[f'{name}:{module.__class__.__name__}'] = module

    return log_end


def is_ml_operation(module):
    """
    This function checks if any given module is of a type that
    we want to analyse for E_ML operations
    """

    e_ml_operations = [nn.Linear, nn.LayerNorm, nn.Embedding, nn.BatchNorm1d,
                       nn.Conv1d, nn.MaxPool1d, nn.AvgPool1d, nn.LSTM, nn.Tanh]

    for e_ml_op in e_ml_operations:
        if isinstance(module, e_ml_op):
            return True
    return False


def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    torch.set_grad_enabled(False)
    return model


def get_all_operations(model_name):
    """
    This function returns the class names of all operations used in a model
    """

    model = load_model(model_name)

    all_operations = set()

    for (name, module) in model.named_modules():
        mname = module.__class__.__name__
        all_operations.add(mname)

    return all_operations


def calibrate_e_ml(model_name, batch_size, input_len, device):
    """
    This function returns information about all the ML level operations in a model
    """
    model = load_model(model_name)
    model = model.eval().to(device)
    for (name, module) in model.named_modules():
        if is_ml_operation(module):
            module.register_forward_pre_hook(log_start_builder(name))
            module.register_forward_hook(log_end_builder(name))

    inputs = torch.randint(1000, size=(batch_size, input_len)).long()
    inputs = inputs.to(device)

    if 't5' in model_name:
        labels = torch.randint(1000, size=(batch_size, input_len)).long()
        labels = labels.to(device)
        _ = model(input_ids=inputs, decoder_input_ids=labels, return_dict=True)
    else:
        _ = model(input_ids=inputs, return_dict=True)

    information = defaultdict(list)
    for module_name in start_times.keys():
        module_info = {'name': module_name, 'module': modules[module_name],
                       'inputs': module_inputs[module_name],
                       'outputs': module_outputs[module_name],
                       'runtime': end_times[module_name] - start_times[
                           module_name]}

        module_identifier = module_name.split(':')[-1]
        information[module_identifier.lower()].append(module_info)

    return information


def main(args):
    operation_names = get_all_operations(args.model_name)
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not cuda_available 
                                    else "cpu")
    information = calibrate_e_ml(args.model_name, args.batch_size,
                                 args.input_len, device)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    args = parse_args()
    main(args)
