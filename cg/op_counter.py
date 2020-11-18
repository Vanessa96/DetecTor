#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FLOPS tutorial (slide 4):
https://www.stat.cmu.edu/~ryantibs/convexopt-F18/lectures/num-lin-alg.pdf

approximate flops for measuring algorithm complexity is still very useful,
but for energy estimation features, the more accurate the better.

reference implementations:
https://github.com/zhijian-liu/torchprofile/blob/master/torchprofile/handlers.py
https://github.com/Swall0w/torchstat/blob/master/torchstat/compute_flops.py
https://github.com/adityaiitb/pyprof2/blob/master/pyprof2/prof/
https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/jit_handles.py
"""

import math

op_counters = {
    'aten::Int': 'aten_Int',
    'aten::ScalarImplicit': 'aten_ScalarImplicit',
    'aten::abs': 'aten_abs',
    'aten::add': 'aten_add',
    'aten::add_': 'aten_add_',
    'aten::addmm': 'aten_addmm',
    'aten::arange': 'aten_arange',
    'aten::bmm': 'aten_bmm',
    'aten::cat': 'aten_cat',
    'aten::clone': 'aten_clone',
    'aten::constant_pad_nd': 'aten_constant_pad_nd',
    'aten::contiguous': 'aten_contiguous',
    'aten::copy_': 'aten_copy_',
    'aten::cumsum': 'aten_cumsum',
    'aten::div': 'aten_div',
    'aten::dropout': 'aten_dropout',
    'aten::einsum': 'aten_einsum',
    'aten::embedding': 'aten_embedding',
    'aten::eq': 'aten_eq',
    'aten::expand_as': 'aten_expand_as',
    'aten::fill_': 'aten_fill_',
    'aten::floor_divide': 'aten_floor_divide',
    'aten::full_like': 'aten_full_like',
    'aten::gather': 'aten_gather',
    'aten::gelu': 'aten_gelu',
    'aten::index': 'aten_index',
    'aten::layer_norm': 'aten_layer_norm',
    'aten::le': 'aten_le',
    'aten::log': 'aten_log',
    'aten::lt': 'aten_lt',
    'aten::masked_fill': 'aten_masked_fill',
    'aten::masked_fill_': 'aten_masked_fill_',
    'aten::matmul': 'aten_matmul',
    'aten::max': 'aten_max',
    'aten::mean': 'aten_mean',
    'aten::min': 'aten_min',
    'aten::mul': 'aten_mul',
    'aten::mul_': 'aten_mul_',
    'aten::ne': 'aten_ne',
    'aten::neg': 'aten_neg',
    'aten::ones': 'aten_ones',
    'aten::permute': 'aten_permute',
    'aten::pow': 'aten_pow',
    'aten::relu': 'aten_relu',
    'aten::repeat': 'aten_repeat',
    'aten::reshape': 'aten_reshape',
    'aten::rsub': 'aten_rsub',
    'aten::select': 'aten_select',
    'aten::size': 'aten_size',
    'aten::slice': 'aten_slice',
    'aten::softmax': 'aten_softmax',
    'aten::split': 'aten_split',
    'aten::sqrt': 'aten_sqrt',
    'aten::squeeze': 'aten_squeeze',
    'aten::stack': 'aten_stack',
    'aten::sub': 'aten_sub',
    'aten::sum': 'aten_sum',
    'aten::t': 'aten_t',
    'aten::tanh': 'aten_tanh',
    'aten::to': 'aten_to',
    'aten::transpose': 'aten_transpose',
    'aten::triu': 'aten_triu',
    'aten::type_as': 'aten_type_as',
    'aten::unsqueeze': 'aten_unsqueeze',
    'aten::view': 'aten_view',
    'aten::where': 'aten_where',
    'aten::zeros': 'aten_zeros',
    'aten::zeros_like': 'aten_zeros_like',
    'prim::Constant': 'prim_Constant',
    'prim::GetAttr': 'prim_GetAttr',
    'prim::ListConstruct': 'prim_ListConstruct',
    'prim::ListUnpack': 'prim_ListUnpack',
    'prim::NumToTensor': 'prim_NumToTensor',
    'prim::TupleConstruct': 'prim_TupleConstruct',
    'prim::TupleUnpack': 'prim_TupleUnpack',
}


def count_flops_io(node):
    op_func = globals().get(op_counters[node.op], None)
    if op_func is None:
        return 0, 0
    else:
        return op_func(node)


def aten_activation(node):
    # todo: linear, relu, tanh, sigmoid, gelu, mish, swish, silu
    pass


def aten_addmm(node):  # also for conv1d in huggingface lib
    # [n, p] = aten::addmm([n, p], [n, m], [m, p], *, *)
    n, m = node.inputs[1].shape
    m, p = node.inputs[2].shape
    return n * m * p


def aten_elementwise(node):
    # todo: add, add_, div, mul, rsub
    pass


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


def aten_softmax(node):
    """ DxVx2 from the NeurIPS 2018 paper below
    "Navigating with graph representations for fast and scalable decoding of neural language models."
    """
    # todo:
    pass
