# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger
from typing import Optional

from fusion_attention_unet import FusionAttentionUnet
from fusion_biassplitgelu import FusionBiasSplitGelu
from fusion_group_norm import FusionGroupNorm
from fusion_options import FusionOptions
from fusion_utils import FusionUtils
from onnx import ModelProto, TensorProto, helper, numpy_helper
from onnx_model_bert import BertOnnxModel
from onnx_model import OnnxModel

logger = getLogger(__name__)


class UnetOnnxModel(BertOnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize UNet ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model, num_heads=num_heads, hidden_size=hidden_size)

    def preprocess(self):
        self.remove_useless_div()
        pass

    def postprocess(self):
        self.convert_conv_to_nhwc()
        self.prune_graph()

    def remove_useless_div(self):
        """Remove Div by 1"""
        div_nodes = [node for node in self.nodes() if node.op_type == "Div"]
        remove_div_node_count = 0
        nodes_to_remove = []
        for div in div_nodes:
            if self.find_constant_input(div, 1.0) == 1:
                nodes_to_remove.append(div)

        for node in nodes_to_remove:
            self.replace_input_of_all_nodes(node.output[0], node.input[0])

        self.remove_nodes(nodes_to_remove)
        logger.info("Removed %d useless Div (by 1) nodes", len(nodes_to_remove))

    def convert_conv_to_nhwc(self):
        conv_to_nhwc_conv = FusionNhwcConv(self)
        conv_to_nhwc_conv.apply()


    def optimize(self, options: Optional[FusionOptions] = None):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_group_norm:
            group_norm_fusion = FusionGroupNorm(self)
            group_norm_fusion.apply()

        if (options is None) or options.enable_bias_splitgelu:
            bias_split_gelu_fusion = FusionBiasSplitGelu(self)
            bias_split_gelu_fusion.apply()

        if (options is None) or options.enable_attention:
            self_attention_fusion = FusionAttentionUnet(self, self.hidden_size, self.num_heads, False, False)
            self_attention_fusion.apply()

            enable_packed_kv = (options is None) or options.enable_packed_kv
            cross_attention_fusion = FusionAttentionUnet(self, self.hidden_size, self.num_heads, True, enable_packed_kv)
            cross_attention_fusion.apply()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm()

        self.fuse_shape()

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()

        self.postprocess()

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        if options is not None and options.enable_gelu_approximation:
            self.gelu_approximation()

        self.remove_unused_constant()

        logger.info(f"opset version: {self.get_opset_version()}")
