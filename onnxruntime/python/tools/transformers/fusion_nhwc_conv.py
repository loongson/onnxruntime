# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import List

from fusion_base import Fusion
from onnx import TensorProto, helper, numpy_helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionNhwcConv(Fusion):
    """Convert Conv to NhwcConv"""

    def __init__(self, model: OnnxModel):
        super().__init__(model, "NhwcConv", ["Conv"], "NhwcConv")

    def create_transpose_node(self, input_name: str, perm: List[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")

        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name

        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])

        return transpose_node

    def fuse(self, conv, input_name_to_nodes, output_name_to_node):
        # Make sure the weights is 4D
        weight_tensor = self.model.get_initializer(conv.input[1])
        if weight_tensor is None:
            return
        weight = numpy_helper.to_array(weight_tensor)
        if len(weight.shape) != 4:
            return

        # Transpose weights from NCHW to NHWC
        weight = weight.transpose(0, 2, 3, 1)

        # Add Transpose node to convert input from NCHW to NHWC
        input_transpose_node = self.create_transpose_node(conv.input[0], [0, 2, 3, 1])

        nhwc_conv_input = input_transpose_node.output[0]

        # Create a tensor for transposed weights (already in NHWC format).
        node_name = self.model.create_node_name("NhwcConv")

        weight_name = node_name + "_weight_NHWC"
        nhwc_weight = helper.make_tensor(
            name=weight_name,
            data_type=TensorProto.FLOAT,
            dims=list(weight.shape),
            vals=weight.flatten().tolist(),
        )
        self.model.add_initializer(nhwc_weight, self.this_graph_name)

        nhwc_output_name = node_name + "_out" + "-" + conv.output[0]
        nhwc_conv = helper.make_node(
            "NhwcConv",
            inputs=[nhwc_conv_input, weight_name] + conv.input[2:],
            outputs=[nhwc_output_name],
            name=node_name + "-" + conv.name,
        )
        nhwc_conv.attribute.extend(conv.attribute)
        nhwc_conv.domain = "com.microsoft"

        output_transpose_node = self.create_transpose_node(nhwc_conv.output[0], [0, 3, 1, 2], conv.output[0])

        self.nodes_to_remove.append(conv)
        self.nodes_to_add.extend([input_transpose_node, nhwc_conv, output_transpose_node])
        self.node_name_to_graph_name[input_transpose_node.name] = self.this_graph_name
        self.node_name_to_graph_name[output_transpose_node.name] = self.this_graph_name
        self.node_name_to_graph_name[nhwc_conv.name] = self.this_graph_name

        self.increase_counter("NhwcConv")
