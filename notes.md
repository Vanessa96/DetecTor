Installation:

[pyenv](https://github.com/pyenv/pyenv) and [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv)
cuda 10.2 cudnn 8.0.3

`pyenv virtualenv 3.7.9 nrg`

`pyenv activate nrg`

`pip install -r requirements.txt`


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

resources information:
- cpu, mem util: https://man7.org/linux/man-pages/man5/proc.5.html
[cpu util impl](https://www.kgoettler.com/post/proc-stat/)
[mem util impl](https://github.com/rfjakob/earlyoom/blob/master/meminfo.c)

- gpu util, gpu mem util, gpu power: 
  [doc](https://docs.nvidia.com/deploy/nvml-api/index.html)
  [api](https://github.com/NVIDIA/nvidia-settings/blob/master/src/nvml.h)

- rapl, CPU/DRAM power: 
  [rapl doc (chapter 14.9)](https://www.intel.com/content/www/us/en/architecture-and-technology/64-ia-32-architectures-software-developer-vol-3b-part-2-manual.html)
  [rapl project](http://web.eece.maine.edu/~vweaver/projects/rapl/index.html) and [rapl-read](https://github.com/deater/uarch-configure/blob/master/rapl-read/rapl-read.c)
  https://man7.org/linux/man-pages/man4/msr.4.html

ideally, due to the reason [`Each sample period may be between 1 second and 1/6 second`](https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t)
we should use [nvmlDeviceGetSamples](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gb7d2a6d2a9b4584cd985765d1ff46c94)
instead of `nvmlDeviceGetUtilizationRates` and `nvmlDeviceGetPowerUsage`
but in practice, the gpu driver buffer stores ~6 samples per second

```text
    // To represent total power drawn by GPU
    NVML_TOTAL_POWER_SAMPLES        = 0,
    // To represent percent of time during which one or more kernels was
    // executing on the GPU
    NVML_GPU_UTILIZATION_SAMPLES    = 1,
    // To represent percent of time during which global (device) memory was
    // being read or written
    NVML_MEMORY_UTILIZATION_SAMPLES = 2,
    // To represent percent of time during which NVENC remains busy
    NVML_ENC_UTILIZATION_SAMPLES    = 3,
    // To represent percent of time during which NVDEC remains busy
    NVML_DEC_UTILIZATION_SAMPLES    = 4,
    // To represent processor clock samples
    NVML_PROCESSOR_CLK_SAMPLES      = 5,
    // To represent memory clock samples
    NVML_MEMORY_CLK_SAMPLES         = 6,
```
