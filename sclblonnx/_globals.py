# Global variables for the sclblonnx package.
import os


# control printing:
SILENT: bool = False  # Boolean indicating whether user feedback should be suppressed.

# Dictionary containing details to check support
VERSION_INFO_LOCATION: str = os.path.dirname(os.path.realpath(__file__)) + "/supported_onnx.json"
ONNX_VERSION_INFO: dict = {}

# Optimizer passes:
OPTIMIZER_PASSES = ['eliminate_deadend',
          'eliminate_duplicate_initializer',
          'eliminate_identity',
          'eliminate_if_with_const_cond',
          'eliminate_nop_cast',
          'eliminate_nop_dropout',
          'eliminate_nop_flatten',
          'eliminate_nop_monotone_argmax',
          'eliminate_nop_pad',
          'eliminate_nop_transpose',
          'eliminate_unused_initializer',
          'extract_constant_to_initializer',
          'fuse_add_bias_into_conv',
          'fuse_bn_into_conv',
          'fuse_consecutive_concats',
          'fuse_consecutive_log_softmax',
          'fuse_consecutive_reduce_unsqueeze',
          'fuse_consecutive_squeezes',
          'fuse_consecutive_transposes',
          'fuse_matmul_add_bias_into_gemm',
          'fuse_pad_into_conv',
          'fuse_transpose_into_gemm',
          'lift_lexical_references']