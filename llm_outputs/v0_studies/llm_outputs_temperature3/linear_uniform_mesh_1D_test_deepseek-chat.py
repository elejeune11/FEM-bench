def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 0.5, 1.0])
    expected_elements = np.array([[0, 1], [1, 2]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (3,)
    assert element_connectivity.shape == (2, 2)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 1.0])
    expected_elements = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)