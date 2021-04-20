from sclblonnx import empty_graph, list_outputs, add_output, rename_output, replace_output, delete_output


def test_list_outputs():
    g = empty_graph()
    assert list_outputs(g), "No outputs listed."
    g = add_output(g, "test", "FLOAT", [0])
    list_outputs(g)
    assert not list_outputs(False), "List outputs should be false."


def test_add_output():
    g = empty_graph()
    g = add_output(g, "test", "FLOAT", [0])
    name = getattr(g.output[0], "name", False)  # get the first output name:
    assert name == "test", "'test' should be in list of outputs."


def test_rename_output():
    g = empty_graph()
    g = add_output(g, "test", "FLOAT", [0])
    g = rename_output(g, "test", "new_name")
    name = getattr(g.output[0], "name", False)  # get the first output name:
    assert name == "new_name", "New name should be in list of outputs."


def test_replace_output():
    g = empty_graph()
    g = add_output(g, "test", "FLOAT", [0])
    g = replace_output(g, "test", "FLOAT", [10,10])

    type = getattr(g.output[0], "type", False)  # get the output type
    ttype = getattr(type, "tensor_type", False)
    shape = getattr(ttype, "shape", False)
    dim = getattr(shape, "dim", False)
    dim_val = getattr(dim[0], "dim_value", False)

    assert dim_val == 10, "New dimension should be 10"


def test_delete_output():
    g = empty_graph()
    g = add_output(g, "test", "FLOAT", [0])
    g = delete_output(g, "test")
    assert len(g.output) == 0, "There should not be any outputs after delete."