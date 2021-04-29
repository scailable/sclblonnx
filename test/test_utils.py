from sclblonnx import empty_graph, add_output, add_input
from sclblonnx.utils import _parse_element, _value, _input_details, _output_details, _print, _load_version_info, \
    _data_type, _data_string
from sclblonnx._globals import ONNX_VERSION_INFO

def test__parse_element():
    g = empty_graph()
    dims = [4, 3, 7]
    g = add_output(g, 'sum', "FLOAT", dims)
    for elem in g.output:
        _, elem_type, _ = _parse_element(elem)
    assert elem_type == "FLOAT", "Element not properly parsed."


def test__value():
    v = _value("test_value", "FLOAT", [1,2,3])
    v_name = getattr(v, "name", False)
    assert v_name ==  "test_value", "Wrong value created."


def test__input_details():
    g = empty_graph()
    g = add_input(g, 'sum', "FLOAT", [4, 3, 7])
    in_det = _input_details(g)
    assert in_det['sum']['data_type'] == "FLOAT", "Input details not correct."


def test__output_details():
    g = empty_graph()
    g = add_output(g, 'sum', "FLOAT", [4, 3, 7])
    out_det = _output_details(g)
    assert out_det['sum']['data_type'] == "FLOAT", "Output details not correct."


def test__print():
    print("\n")
    _print("Red warning.")
    _print("Normal feedback", "MSG")
    _print("Green literal", "LIT")
    pass


def test__load_version_info():
    assert not ONNX_VERSION_INFO, "Should not be loaded."
    _load_version_info()
    assert not ONNX_VERSION_INFO, "Should be loaded."


def test__data_type():
    assert _data_type("FLOAT") == 1, "Float should be 1."
    assert not _data_type("BLA"), "Bla should not be a data type."


def test__data_string():
    assert _data_string(1) == "FLOAT", "Float should be 1."
    assert not _data_string(99), "99 should not be a data string."