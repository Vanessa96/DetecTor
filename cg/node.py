from __future__ import annotations
from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass
class OpNode:
    # op: str
    id: str  # use output node debugName
    scope: str  # scope + id
    op: str  # operation type: matmul, add, mul, div, etc.
    inputs: list[DataNode]
    outputs: list[DataNode]
    # attr: dict[str, Any] = field(default_factory=dict)  # extra information
    # flops: int = 0  # number of operations


@dataclass
class DataNode:
    id: str
    dtype: str
    shape: list[int]  # tensor shape
    # params: bool = False  # flattened array weights
    # attr: dict[str, Any] = field(default_factory=dict)  # extra information
    # mem_read_bytes: int = 0  # amount of data read from input nodes
    # mem_write_bytes: int = 0  # amount of data written to output nodes


@dataclass
class Module:
    id: str  # scope + id
    nodes: list[OpNode]


@dataclass
class Graph:
    name: str
    nodes: list[OpNode]
    inputs: list[DataNode]
    outputs: list[DataNode]
    # attr: dict[str, Any] = field(default_factory=dict)  # extra information


from torch import nn


class LinearModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, x):
        y = self.dense(x)
        return y


# from transformers.activations import ACT2FN

# class BertIntermediate(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
#         if isinstance(config.hidden_act, str):
#             self.intermediate_act_fn = ACT2FN[config.hidden_act]
#         else:
#             self.intermediate_act_fn = config.hidden_act
#
#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.intermediate_act_fn(hidden_states)
#         return hidden_states


if __name__ == '__main__':
    import torch
    from transformers import BertModel

    # from transformers.modeling_bert import BertIntermediate

    inputs = torch.randint(1000, size=(1, 100)).long()
    model = BertModel.from_pretrained("prajjwal1/bert-tiny", torchscript=True)

    trace = torch.jit.trace(model, inputs)
    fc_graph = trace.inlined_graph
    config = model.config
    config.hidden_act = 'gelu_fast'
    # fc_model = BertIntermediate(config)
    # fc_model = LinearModel(config)
    # input_len = 32
    # input_states = torch.rand((input_len, config.hidden_size))
    # fc_trace = torch.jit.trace(fc_model, (input_states,))
    # fc_graph = fc_trace.inlined_graph
    fc_input_nodes = {i.debugName(): i for i in fc_graph.inputs()}
    fc_output_nodes = {i.debugName(): i for i in fc_graph.outputs()}
    data_nodes = {}
    nodes = []
    gi_nodes = []
    go_nodes = []
    ops = set()
    for n in fc_graph.nodes():
        node_inputs = list(n.inputs())
        node_outputs = list(n.outputs())
        node_id = node_outputs[0].debugName()
        node_scope = n.scopeName()
        node_op = n.kind()
        ops.add(node_op)
        in_nodes = []
        for ni in node_inputs:
            if isinstance(ni.type(), torch.TensorType):
                dtype = ni.type().scalarType()  # Float
                shape = ni.type().sizes()  # list of int
                # TODO: check how to detect if this is a parameter data node
                ni_node = DataNode(ni.debugName(), dtype, shape)
                in_nodes.append(ni_node)
                if ni.debugName() in fc_input_nodes:
                    gi_nodes.append(ni_node)
            else:
                # need to handle int, int[], Long(), Device, bool, None
                pass
        no_nodes = []
        for no in node_outputs:
            if isinstance(no.type(), torch.TensorType):
                dtype = no.type().scalarType()  # Float
                shape = no.type().sizes()  # list of int
                # TODO: check how to detect if this is a parameter data node
                no_node = DataNode(no.debugName(), dtype, shape)
                no_nodes.append(no_node)
                if no.debugName() in fc_output_nodes:
                    go_nodes.append(no_node)
            else:
                # need to handle int, int[], Long(), Device, bool, None
                pass
        op_node = OpNode(node_id, node_scope, node_op, in_nodes, no_nodes)
        nodes.append(op_node)
    graph = Graph('bert-tiny', nodes, gi_nodes, go_nodes)
    print(graph)
    print(ops)

