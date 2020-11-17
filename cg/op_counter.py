#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Qingqing Cao, https://awk.ai/, Twitter@sysnlp"

import math

op_counters = {
    'aten::Int': 'aten_Int',
    'aten::add': 'aten_add',
    'aten::add_': 'aten_add_',
    'aten::addmm': 'aten_addmm',
    'aten::contiguous': 'aten_contiguous',
    'aten::div': 'aten_div',
    'aten::dropout': 'aten_dropout',
    'aten::embedding': 'aten_embedding',
    'aten::layer_norm': 'aten_layer_norm',
    'aten::matmul': 'aten_matmul',
    'aten::mul': 'aten_mul',
    'aten::ones': 'aten_ones',
    'aten::permute': 'aten_permute',
    'aten::rsub': 'aten_rsub',
    'aten::select': 'aten_select',
    'aten::size': 'aten_size',
    'aten::slice': 'aten_slice',
    'aten::softmax': 'aten_softmax',
    'aten::t': 'aten_t',
    'aten::tanh': 'aten_tanh',
    'aten::to': 'aten_to',
    'aten::transpose': 'aten_transpose',
    'aten::unsqueeze': 'aten_unsqueeze',
    'aten::view': 'aten_view',
    'aten::zeros': 'aten_zeros',
    'prim::Constant': 'prim_Constant',
    'prim::GetAttr': 'prim_GetAttr',
    'prim::ListConstruct': 'prim_ListConstruct',
    'prim::NumToTensor': 'prim_NumToTensor',
    'prim::TupleConstruct': 'prim_TupleConstruct'
}


def count_flops_io(node):
    op_func = globals().get(op_counters[node.op], None)
    if op_func is None:
        return 0, 0
    else:
        return op_func(node)


def aten_addmm(node):
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p

def aten_layer_norm(node):
    os = node.outputs[0].shape
    return math.prod(os)

def aten_mul(node):
    os = node.outputs[0].shape
    return math.prod(os)


def aten_matmul(node):
    if node.inputs[0].ndim == 1 and node.inputs[1].ndim == 1:
        # [] = aten::matmul([n], [n])
        n = node.inputs[0].shape[0]
        return n
    elif node.inputs[0].ndim == 1 and node.inputs[1].ndim == 2:
        # [m] = aten::matmul([n], [n, m])
        n, m = node.inputs[1].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 1:
        # [n] = aten::matmul([n, m], [m])
        n, m = node.inputs[0].shape
        return n * m
    elif node.inputs[0].ndim == 2 and node.inputs[1].ndim == 2:
        # [n, p] = aten::matmul([n, m], [m, p])
        n, m = node.inputs[0].shape
        m, p = node.inputs[1].shape
        return n * m * p
    elif node.inputs[0].ndim == 1:
        # [..., m] = aten::matmul([n], [..., n, m])
        *b, n, m = node.inputs[1].shape
        return math.prod(b) * n * m
    elif node.inputs[1].ndim == 1:
        # [..., n] = aten::matmul([..., n, m], [m])
        *b, n, m = node.inputs[0].shape
        return math.prod(b) * n * m
    else:
        # [..., n, p] = aten::matmul([..., n, m], [..., m, p])
        *b, n, p = node.outputs[0].shape
        *_, n, m = node.inputs[0].shape
        *_, m, p = node.inputs[1].shape
        return math.prod(b) * n * m * p
