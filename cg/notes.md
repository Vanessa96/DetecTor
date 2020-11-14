
```text
# sample jit traced graph
graph(%self.1 : __torch__.___torch_mangle_3.BertIntermediate,
      %input : Float(32:128, 128:1)):
  %2 : __torch__.torch.nn.modules.linear.___torch_mangle_2.Linear = prim::GetAttr[name="dense"](%self.1)
  %4 : int = prim::Constant[value=1](), scope: __module.dense # torch/nn/functional.py:1674:0
  %5 : Tensor = prim::GetAttr[name="bias"](%2)
  %6 : Tensor = prim::GetAttr[name="weight"](%2)
  %7 : Float(128:1, 512:128) = aten::t(%6), scope: __module.dense # torch/nn/functional.py:1674:0
  %8 : Float(32:512, 512:1) = aten::addmm(%5, %input, %7, %4, %4), scope: __module.dense # torch/nn/functional.py:1674:0
  return (%8)

```

understanding data structure of torch jit traced graph:
[jit overview doc](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/OVERVIEW.md)
[jit ir](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/ir/ir.cpp)
[jit operator](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/runtime/operator.cpp)
[jit op doc](https://pytorch.org/docs/master/jit_builtin_functions.html)
[jit interpret graph](https://pytorch.org/docs/stable/jit.html#interpreting-graphs)
[scope info](https://github.com/pytorch/pytorch/pull/3016/files)

nodes and values
each node has 0, 1 or multiple input values and one output value
each value is a access to tensor or tensor itself, unique id, value.debugName
each value tensor, dtype, shape ,size
  dtype: Tensor or scalar, int, float etc.
  
prim::Return
prim::Param

kind = prim:: or aten::
node: op (n.kind()), scope (n.scopeName()), id (n.output().debugName())
  so node can use output value id

only track 
    data nodes (at least one tensor in input or output) and 
    op nodes that are compute (exclude or merge access/property nodes)

graph, inputs and outputs data nodes

