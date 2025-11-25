def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert np.allclose(node_coords, [0.0, 0.5, 1.0])
    assert np.array_equal(element_connectivity, [[0, 1], [1, 2]])

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert np.allclose(node_coords, [0.0, 1.0])
    assert np.array_equal(element_connectivity, [[0, 1]])