from sclblonnx import empty_graph, list_inputs, add_input, rename_input, replace_input, delete_input


def test_list_inputs():
    g = empty_graph()
    assert list_inputs(g), "No inputs listed."
    g = add_input(g, "test", "FLOAT", [0])
    list_inputs(g)
    assert not list_inputs(False), "List inputs should be false."


def test_add_input():
    g = empty_graph()
    g = add_input(g, "test", "FLOAT", [0])
    name = getattr(g.input[0], "name", False)  # get the first input name:
    assert name == "test", "'test' should be in list of inputs."


def test_rename_input():
    g = empty_graph()
    g = add_input(g, "test", "FLOAT", [0])
    g = rename_input(g, "test", "new_name")
    name = getattr(g.input[0], "name", False)  # get the first input name:
    assert name == "new_name", "New name should be in list of inputs."


def test_replace_input():
    g = empty_graph()
    g = add_input(g, "test", "FLOAT", [0])
    g = replace_input(g, "test", "FLOAT", [10,10])

    type = getattr(g.input[0], "type", False)  # get the input type
    ttype = getattr(type, "tensor_type", False)
    shape = getattr(ttype, "shape", False)
    dim = getattr(shape, "dim", False)
    dim_val = getattr(dim[0], "dim_value", False)

    assert dim_val == 10, "New dimension should be 10"


def test_delete_input():
    g = empty_graph()
    g = add_input(g, "test", "FLOAT", [0])
    g = delete_input(g, "test")
    assert len(g.input) == 0, "There should not be any inputs after delete."