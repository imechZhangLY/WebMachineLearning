from onnx import TensorProto, OperatorSetIdProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model
import onnx

op = OperatorSetIdProto()
op.version = 9
# inputs
M = 4000
N = 4000
# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
A = make_tensor_value_info('A', TensorProto.FLOAT, [M, N])
B = make_tensor_value_info('B', TensorProto.FLOAT, [N, M])

# outputs, the shape is left undefined

C = make_tensor_value_info('C', TensorProto.FLOAT, [M, M])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node = make_node('MatMul', ['A', 'B'], ['C'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node],  # nodes
                    'mat_mul',  # a name
                    [A, B],  # inputs
                    [C])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph, opset_imports=[op])

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

onnx.save(onnx_model, "mat_mul.onnx")