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

# ops marked with # p1 are implemented
# ops marked with # only are planned with low priority
import warnings

op_counters = {
    'aten::Int': 'aten_Int',
    'aten::ScalarImplicit': 'aten_ScalarImplicit',
    'aten::abs': 'aten_abs',  #
    'aten::add': 'aten_add',  # p1
    'aten::add_': 'aten_add_',  # p1
    'aten::addmm': 'aten_addmm',  # p1
    'aten::arange': 'aten_arange',  #
    'aten::bmm': 'aten_bmm',  # p1
    'aten::cat': 'aten_cat',  #
    'aten::clone': 'aten_clone',  # p1
    'aten::constant_pad_nd': 'aten_constant_pad_nd',  #
    'aten::contiguous': 'aten_contiguous',
    'aten::copy_': 'aten_copy_',  # p1
    'aten::cumsum': 'aten_cumsum',  #
    'aten::div': 'aten_div',  # p1
    'aten::dropout': 'aten_dropout',
    'aten::einsum': 'aten_einsum',  # p1
    'aten::embedding': 'aten_embedding',  # p1
    'aten::eq': 'aten_eq',  #
    'aten::expand_as': 'aten_expand_as',
    'aten::fill_': 'aten_fill_',  #
    'aten::floor_divide': 'aten_floor_divide',  #
    'aten::full_like': 'aten_full_like',  #
    'aten::gather': 'aten_gather',  #
    'aten::gelu': 'aten_gelu',  # p1
    'aten::index': 'aten_index',  #
    'aten::layer_norm': 'aten_layer_norm',  # p1
    'aten::le': 'aten_le',  #
    'aten::log': 'aten_log',  #
    'aten::lt': 'aten_lt',  #
    'aten::masked_fill': 'aten_masked_fill',
    'aten::masked_fill_': 'aten_masked_fill_',
    'aten::matmul': 'aten_matmul',  # p1
    'aten::max': 'aten_max',  #
    'aten::mean': 'aten_mean',  #
    'aten::min': 'aten_min',  #
    'aten::mul': 'aten_mul',  # p1
    'aten::mul_': 'aten_mul_',  # p1
    'aten::ne': 'aten_ne',  #
    'aten::neg': 'aten_neg',  #
    'aten::ones': 'aten_ones',  #
    'aten::permute': 'aten_permute',  #
    'aten::pow': 'aten_pow',  # p1
    'aten::relu': 'aten_relu',  # p1
    'aten::repeat': 'aten_repeat',  #
    'aten::reshape': 'aten_reshape',  #
    'aten::rsub': 'aten_rsub',  # p1
    'aten::select': 'aten_select',  #
    'aten::size': 'aten_size',
    'aten::slice': 'aten_slice',  #
    'aten::softmax': 'aten_softmax',  # p1
    'aten::split': 'aten_split',  #
    'aten::sqrt': 'aten_sqrt',  # p1
    'aten::squeeze': 'aten_squeeze',  #
    'aten::stack': 'aten_stack',  #
    'aten::sub': 'aten_sub',  # p1
    'aten::sum': 'aten_sum',  # p1
    'aten::t': 'aten_t',  #
    'aten::tanh': 'aten_tanh',  # p1
    'aten::to': 'aten_to',
    'aten::transpose': 'aten_transpose',  #
    'aten::triu': 'aten_triu',  #
    'aten::type_as': 'aten_type_as',
    'aten::unsqueeze': 'aten_unsqueeze',  #
    'aten::view': 'aten_view',
    'aten::where': 'aten_where',  #
    'aten::zeros': 'aten_zeros',  #
    'aten::zeros_like': 'aten_zeros_like',  #
    'prim::Constant': 'prim_Constant',  # need unpacking
    'prim::GetAttr': 'prim_GetAttr',
    'prim::ListConstruct': 'prim_ListConstruct',  # need unpacking
    'prim::ListUnpack': 'prim_ListUnpack',
    'prim::NumToTensor': 'prim_NumToTensor',
    'prim::TupleConstruct': 'prim_TupleConstruct',  # need unpacking
    'prim::TupleUnpack': 'prim_TupleUnpack',
}


def _dtype_to_bytes(dtype):
    dtype = dtype.lower()
    if dtype in {"uint8", "int8", "byte", "char", 'bool'}:
        return 1
    elif dtype in {"float16", "half", "int16", "short"}:
        return 2
    elif dtype in {"float32", "float", "int32", "int"}:
        return 4
    elif dtype in {"int64", "long", "float64", "double"}:
        return 8
    elif dtype in {'none'}:
        return 0
    else:
        warnings.warn(f"unknown data type (will return 0): {dtype}!")
        # handle List[int], Device
        return 0


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
    n_1 = node.inputs[1]
    n, m = n_1.shape
    # n_2 = node.inputs[2] # n_2 maybe empty, infer from outputs
    p = node.outputs[0].shape[-1]
    flops = n * m * p * 2
    mem_bytes = (m * n + m * p + n * p) * _dtype_to_bytes(n_1.dtype)
    return flops, mem_bytes


def aten_bmm(node):
    # [b, n, p] = aten::bmm([b, m, n], [b, n, p], *, *)
    n_0 = node.inputs[0]
    n_1 = node.inputs[1]
    b, m, n = n_0.shape
    p = n_1.shape[-1]
    assert b == n_1.shape[0] and n == n_1.shape[1]
    flops = b * n * m * p * 2
    mem_bytes = b * (m * n + m * p + n * p) * _dtype_to_bytes(n_1.dtype)
    return flops, mem_bytes


def aten_clone(node):
    in_0 = node.inputs[0]
    in_shape = in_0.shape
    mem_bytes = 2 * math.prod(in_shape) * _dtype_to_bytes(in_0.dtype)
    return 0, mem_bytes


def aten_copy_(node):
    in_0 = node.inputs[0]
    in_vol = math.prod(in_0.shape)
    out_0 = node.outputs[0]
    out_vol = math.prod(out_0.shape)
    mem_bytes = (in_vol + out_vol) * _dtype_to_bytes(in_0.dtype)
    return 0, mem_bytes


def aten_einsum(node):
    # todo: albert torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
    #  check reference impl
    #  https://github.com/facebookresearch/fvcore/blob/7bc26c1f2a3ebc1ff91f266bf3ac37b23ee35842/fvcore/nn/jit_handles.py#L255
    pass


def aten_embedding(node):
    return 0


def aten_gelu(node):
    return 0


def aten_relu(node):
    return 0


def aten_unary(fops_per_elem=1, mem_per_elem=1):
    # for: abs, rsqrt
    def counter(node):
        in_0 = node.inputs[0]
        in_shape = in_0.shape
        vol = math.prod(in_shape)
        flops = vol * fops_per_elem
        mem_bytes = vol * mem_per_elem * _dtype_to_bytes(in_0.dtype)
        return flops, mem_bytes

    return counter


# todo: support broadcast
#  https://github.com/adityaiitb/pyprof2/blob/b2ac33876a2ab5bbd41595f0692a0fc936e7d8b7/pyprof2/prof/pointwise.py#L97

def aten_binary(fops_per_elem=1, mem_per_elem=2):
    def counter(node):
        # for: add, add_, div, mul, mul_, rsub, sub
        out_0 = node.outputs[0]
        out_shape = out_0.shape
        vol = math.prod(out_shape)
        flops = vol * fops_per_elem
        mem_bytes = vol * mem_per_elem * _dtype_to_bytes(out_0.dtype)
        return flops, mem_bytes

    return counter


aten_abs = aten_unary()
aten_add = aten_binary()
aten_add_ = aten_binary()
aten_div = aten_binary()
aten_mul = aten_binary(mem_per_elem=3)
aten_mul_ = aten_binary(mem_per_elem=3)
aten_pow = aten_binary()
aten_rsub = aten_binary(mem_per_elem=3)
aten_rsqrt = aten_unary(fops_per_elem=2)
aten_softmax = aten_unary(fops_per_elem=5)
aten_sub = aten_binary()
aten_tanh = aten_unary(fops_per_elem=4)


def aten_layer_norm(node):
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


def aten_sum(node):
    return
