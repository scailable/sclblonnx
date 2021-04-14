# Global variables for the sclblonnx package.
import os

# Dictionary containing details to check support
VERSION_INFO_LOCATION: str = os.path.dirname(os.path.realpath(__file__)) + "/supported_onnx.json"
ONNX_VERSION_INFO: dict = {}

# Node counter:
NODE_COUNT = 1

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

# Data types, see https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
DATA_TYPES = {
    "FLOAT": 1,
    "UINT8": 2,
    "INT8": 3,
    "UINT16": 4,
    "INT16": 5,
    "INT32": 6,
    "INT64": 7,
    # "STRING" : 8,
    "BOOL": 9,
    "FLOAT16": 10,
    "DOUBLE": 11,
    "UINT32": 12,
    "UINT64": 13,
    "COMPLEX64": 14,
    "COMPLEX128": 15
}
