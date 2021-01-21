#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Yash Kumar Lal"

import argparse
import time
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers import AutoModel
from cg.node import construct_aggregation_graph
from run_level_exp import construct_aggregation_graph
from ete3 import Tree
import toytree

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.output(x)
        x = nn.Softmax(dim=-1)(x)
        return x

class TreeNode(object):
    def __init__(self, scope, instance_type, level, parent_name, callable_module):
        self.instance_type = instance_type
        self.scope = scope
        self.level = level
        self.parent_name = parent_name
        self.child_nodes = []
        self.callable_module = callable_module

    def __str__(self):
        ret = "(" + self.scope.split('.')[-1] + ","
        for child in self.child_nodes:
            ret += child.__str__()
        ret += ")"
        return ret

    def description(self):
        print(f'NODE INFORMATION - Scope: {self.scope}, instance type: {self.instance_type}, level: {self.level}, parent: {self.parent_name}')

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
    parser.add_argument('--out-file', type=str,
                        help='Newick tree format output file name')
    args, _ = parser.parse_known_args()
    return args

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    torch.set_grad_enabled(False)
    return model

def create_tree_from_modules(model):

    """
    Use the named modules of a model to create a Newick format tree to visualise all the parts of a model
    """

    model_operation_information = []

    for (name, module) in model.named_modules():
        mname = module.__class__.__name__
        if name == '':
            prefix = 'root'
        else:
            prefix = 'root.'
        model_operation_information.append((prefix+name, mname, module))

    for operations in model_operation_information:
        scope = operations[0]
        module = operations[2]
        if scope == 'root':
            parent_name = ''
        else:
            parent_name = '.'.join(scope.split('.')[:-1])

    root_operation = model_operation_information[0]
    root = TreeNode('root', root_operation[1], 0, '', operations[2])

    tree = {}
    tree['root'] = root

    parent_child_nodes = defaultdict(list)
    for operations in model_operation_information[1:]:
        scope = operations[0]
        instance_type = operations[1]
        module = operations[2]
        level = len(scope.split('.')) - 1
        parent_name = '.'.join(scope.split('.')[:-1])
        node = TreeNode(scope, instance_type, level, parent_name, module)
        parent_child_nodes[parent_name].append(node)
        tree[scope] = node

    for name, node in tree.items():
        node.child_nodes = parent_child_nodes[name]

    return root, tree

def run_model(model_name, device, out_file):
    model = load_model(model_name)
    inputs = torch.randint(1000, size=(16, 256)).long()
    # model = TestNet()
    # inputs = torch.randint(1000, size=(16, 256)).float()
    model = model.eval().to(device)
    inputs = inputs.to(device)

    trace = torch.jit.trace(model, inputs)
    trace_graph = trace.inlined_graph
    # graph, _ = construct_aggregation_graph(trace_graph, 'testnet')
    graph, _ = construct_aggregation_graph(trace_graph, model_name)

    # dict1 - id to node
    # dict2 - scope to list of ids
    id_to_node_map = dict()
    scope_to_node_ids_map = defaultdict(list)
    for node in graph.nodes:
        id_to_node_map[node.id] = node
        node_scope = node.scope
        if node.scope == '':
            node_scope = 'root'
        scope_to_node_ids_map[node_scope].append(node.id)

    root, tree = create_tree_from_modules(model)

    viz_tree = toytree.tree(root.__str__()+';', tree_format=8)
    canvas, axes, mark = viz_tree.draw(width=400, height=300)

def main(args):
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not cuda_available 
                                    else "cpu")

    run_model(args.model_name, device, args.out_file)

if __name__ == '__main__':
    args = parse_args()
    main(args)