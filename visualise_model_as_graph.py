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
from graphviz import Digraph
import copy

class TreeNode(object):
    def __init__(self, scope, instance_type, level, parent_name, callable_module):
        self.instance_type = instance_type
        self.scope = scope
        self.level = level
        self.parent_name = parent_name
        self.child_nodes = []
        self.callable_module = callable_module

    def __str__(self):
        ret = "(" + self.scope.split('.')[-1]
        for child in self.child_nodes:
            ret += "," + child.__str__()
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
                        help='Graphviz representation file name')
    args, _ = parser.parse_known_args()
    return args

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.torchscript = True
    model = AutoModel.from_config(config)
    torch.set_grad_enabled(False)
    return model

def graphviz_representation(tree):

    """
    This function creates a graphviz style digraph representation for the model graph
    """

    math_ops_name = ['matmul', 'bmm', 'softmax', 'einsum']
    ml_ops_name = ['Linear', 'LayerNorm', 'Embedding', 'BatchNorm1d', 'Conv1d', 'MaxPool1d', 'AvgPool1d', 'LSTM', 'Tanh', 'Conv1D']

    dot = Digraph(comment='Model Graph')
    node_count = 0
    graphviz_node_id_mapping = {}
    # first create nodes with their labels
    for key, node in tree.items():
        if node.instance_type in ml_ops_name:
            dot.attr('node', color='red', shape='oval')
        elif node.instance_type in math_ops_name:
            dot.attr('node', color='blue', shape='diamond')
        else:
            dot.attr('node', color='black', shape='rectangle')
        dot.node(str(node_count), node.scope.split('.')[-1] + ':' + node.instance_type)
        graphviz_node_id_mapping[node.scope] = str(node_count)
        node_count += 1
    # add edges between nodes using node ids assigned in previous loop
    for key, node in tree.items():
        for child_node in node.child_nodes:
            dot.edge(graphviz_node_id_mapping[node.scope], graphviz_node_id_mapping[child_node.scope])
    return dot

def create_tree_from_modules(model):

    """
    Use the named modules of a model to create a Newick format tree to visualise all the parts of a model
    """

    model_operation_information = []

    # create an iterable list of module information since model.named_modules does not function as a true list
    for (name, module) in model.named_modules():
        mname = module.__class__.__name__
        if name == '':
            prefix = 'root'
        else:
            prefix = 'root.'
        model_operation_information.append((prefix+name, mname, module))

    # add prefix to every node's scope since, by default, the root node of a graph in PyTorch has empty string as scope
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

    # create nodes and keep track of parent-child relationships
    parent_child_nodes = defaultdict(list)
    for operations in model_operation_information[1:]:
        scope = operations[0]
        instance_type = operations[1]
        if instance_type == 'Dropout':
            continue
        module = operations[2]
        level = len(scope.split('.')) - 1
        parent_name = '.'.join(scope.split('.')[:-1])
        node = TreeNode(scope, instance_type, level, parent_name, module)
        parent_child_nodes[parent_name].append(node)
        tree[scope] = node

    # add child information to each node using the information previously stored about parent-child relationships
    for name, node in tree.items():
        node.child_nodes = parent_child_nodes[name]

    return root, tree

def run_model_to_graph(model_name, device, out_file):
    model = load_model(model_name)
    inputs = torch.randint(1000, size=(16, 256)).long()
    model = model.eval().to(device)
    inputs = inputs.to(device)

    trace = torch.jit.trace(model, inputs)
    trace_graph = trace.inlined_graph
    graph, _ = construct_aggregation_graph(trace_graph, model_name)

    # these dictionaries are important when reconciling jit trace to the tree created from modules
    id_to_node_map = dict()
    scope_to_node_ids_map = defaultdict(list)
    for node in graph.nodes:
        id_to_node_map[node.id] = node
        node_scope = node.scope
        if node.scope == '':
            node_scope = 'root'
        scope_to_node_ids_map[node_scope].append(node.id)

    root, tree = create_tree_from_modules(model)

    # for math ops, look into jit trace, create nodes accordingly and add them to correct position in model graph
    for scope, node_ids in scope_to_node_ids_map.items():
        for node_id in node_ids:
            jit_node = id_to_node_map[node_id]
            if jit_node.mem_bytes != 0 or jit_node.flops != 0:
                if jit_node.scope == '':
                    scope_to_match = 'root'
                else:
                    scope_to_match = 'root.' + jit_node.scope
                if jit_node.op not in ['aten::matmul', 'aten::bmm', 'aten::einsum', 'aten::softmax']:
                    continue
                node_in_position = tree[scope_to_match]
                if node_in_position.instance_type in ['Linear', 'LayerNorm', 'Embedding', 'BatchNorm1d', \
                       'Conv1d', 'MaxPool1d', 'AvgPool1d', 'LSTM', 'Tanh', \
                       'Conv1D']:
                    continue
                new_scope_name = scope_to_match + '.' + jit_node.op.split('::')[-1]
                new_node = TreeNode(new_scope_name, jit_node.op.split('::')[-1], node_in_position.level+1, node_in_position.scope, jit_node.op)
                node_in_position.child_nodes.append(new_node)
                tree[new_node.scope] = new_node
    return root, tree

def main(args):
    cuda_exist = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_exist and not cuda_available 
                                    else "cpu")

    root, tree = run_model_to_graph(args.model_name, device, args.out_file)

    dot = graphviz_representation(tree)

    # save graphviz representation to file, then use shell command to generate final graph image
    with open(args.out_file, 'w+') as fp:
        fp.write(dot.source)

    # in shell, run ```dot -Tpdf args.out_file -o final_graph_file.pdf``` to finally generate graph from source code saved above

if __name__ == '__main__':
    args = parse_args()
    main(args)