def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert len(node_coords) == num_elements + 1
    assert hasattr(element_connectivity, 'shape')
    assert element_connectivity.shape == (num_elements, 2)
    nodes = node_coords.tolist()
    tol = 1e-12
    assert abs(nodes[0] - x_min) < tol
    assert abs(nodes[-1] - x_max) < tol
    h = (x_max - x_min) / num_elements
    diffs = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
    for d in diffs:
        assert abs(d - h) < tol
    expected_conn = [[0, 1], [1, 2], [2, 3], [3, 4]]
    assert element_connectivity.tolist() == expected_conn
    for (a, b) in expected_conn:
        assert 0 <= a < len(nodes)
        assert 0 <= b < len(nodes)
        assert b - a == 1

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max, num_elements) = (-2.5, 3.5, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert len(node_coords) == 2
    assert hasattr(element_connectivity, 'shape')
    assert element_connectivity.shape == (1, 2)
    nodes = node_coords.tolist()
    tol = 1e-12
    assert abs(nodes[0] - x_min) < tol
    assert abs(nodes[1] - x_max) < tol
    expected_conn = [[0, 1]]
    assert element_connectivity.tolist() == expected_conn
    (a, b) = expected_conn[0]
    assert 0 <= a < len(nodes)
    assert 0 <= b < len(nodes)
    assert b - a == 1