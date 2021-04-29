import os
import numpy as np
from onnx import onnx_ml_pb2 as xpb2
from sclblonnx import empty_graph, graph_from_file, graph_to_file, run, list_data_types, list_operators, sclbl_input


def test_empty_graph():
    g = empty_graph()
    assert type(g) is xpb2.GraphProto, "Failed to create empty graph."


def test_graph_from_file():
    g = graph_from_file("files/non-existing-file.onnx")
    assert not g, "Graph from file failed to check emtpy file."
    g = graph_from_file("files/example01.onnx")
    assert type(g) is xpb2.GraphProto, "Graph from file failed to open file."


def test_graph_to_file():
    g = empty_graph()
    check1 = graph_to_file(g, "")
    assert not check1, "Graph to file failed should have failed."
    check2 = graph_to_file(g, "files/test_graph_to_file.onnx")
    assert check2, "Graph to file failed to write file."
    os.remove("files/test_graph_to_file.onnx")


def test_run():
    g = graph_from_file("files/add.onnx")
    example = {"x1": np.array([2]).astype(np.float32), "x2": np.array([5]).astype(np.float32)}
    result = run(g,
                    inputs=example,
                    outputs=["sum"]
                    )
    assert result[0] == 7, "Add output not correct."
    result = run(g, inputs="", outputs="sum")
    assert not result, "Model with this input should not run."


def test_display():
    from onnx import TensorProto
    print(TensorProto.DOUBLE)

    return True  # No test for display


def test_scblbl_input():
    example = {"in": np.array([1,2,3,4]).astype(np.int32)}
    result = sclbl_input(example, _verbose=False)
    assert result == '{"input": CAQQBkoQAQAAAAIAAAADAAAABAAAAA==, "type":"pb"}', "PB output not correct."

    example = {"x1": np.array([1,2,3,4]).astype(np.int32), "x2": np.array([1,2,3,4]).astype(np.int32)}
    result = sclbl_input(example, _verbose=False)
    assert result == '{"input": ["CAQQBkoQAQAAAAIAAAADAAAABAAAAA==","CAQQBkoQAQAAAAIAAAADAAAABAAAAA=="], "type":"pb"}',\
        "PB output 2 not correct. "

    example = {"in": np.array([1,2,3,4]).astype(np.int32)}
    result = sclbl_input(example, "raw", _verbose=False)
    assert result == '{"input": AQAAAAIAAAADAAAABAAAAA==, "type":"raw"}', "Raw output not correct."

    example = {"x1": np.array([1,2,3,4]).astype(np.int32), "x2": np.array([1,2,3,4]).astype(np.int32)}
    result = sclbl_input(example, "raw", _verbose=False)
    assert result == '{"input": ["AQAAAAIAAAADAAAABAAAAA==","AQAAAAIAAAADAAAABAAAAA=="], "type":"raw"}',\
        "Raw output 2 not correct. "


def test_list_data_types():
    test = list_data_types()
    assert test, "Data types should be listed."


def test_list_operators():
    test = list_operators()
    assert test, "Operators should be listed."