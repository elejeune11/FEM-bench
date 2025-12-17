def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    import math
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    node_coords, element_connectivity = result
    expected_num_nodes = num_elements + 1
    node_list = [float(v) for v in node_coords]
    assert len(node_list) == expected_num_nodes
    dx = (x_max - x_min) / num_elements
    expected_nodes = [x_min + i * dx for i in range(expected_num_nodes)]
    for a, b in zip(node_list, expected_nodes):
        assert math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    for i in range(expected_num_nodes - 1):
        assert node_list[i] < node_list[i + 1]
        assert math.isclose(node_list[i + 1] - node_list[i], dx, rel_tol=0, abs_tol=1e-12)
    rows = [list(r) for r in element_connectivity]
    assert len(rows) == num_elements
    for i, row in enumerate(rows):
        assert len(row) == 2
        assert row == [i, i + 1]
        assert 0 <= row[0] < expected_num_nodes
        assert 0 < row[1] <= expected_num_nodes - 1

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    import math
    x_min = -2.5
    x_max = 3.5
    num_elements = 1
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, tuple) and len(result) == 2
    node_coords, element_connectivity = result
    node_list = [float(v) for v in node_coords]
    assert len(node_list) == 2
    assert math.isclose(node_list[0], x_min, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(node_list[1], x_max, rel_tol=0, abs_tol=1e-12)
    assert node_list[0] < node_list[1]
    assert math.isclose(node_list[1] - node_list[0], x_max - x_min, rel_tol=0, abs_tol=1e-12)
    rows = [list(r) for r in element_connectivity]
    assert len(rows) == 1
    assert rows[0] == [0, 1]