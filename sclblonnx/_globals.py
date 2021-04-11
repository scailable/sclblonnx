# Global variables for the sclblonnx package.
import os


# control printing:
SILENT: bool = False  # Boolean indicating whether user feedback should be suppressed.

# Dictionary containing details to check support
VERSION_INFO_LOCATION: str = os.path.dirname(os.path.realpath(__file__)) + "/supported_onnx.json"
ONNX_VERSION_INFO: dict = {}
