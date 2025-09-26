def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert node_coords.shape == (5,)
    assert element_connectivity.shape == (4, 2)
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_connectivity)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max, num_elements) = (-1.0, 3.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([-1.0, 3.0])
    expected_connectivity = np.array([[0, 1]])
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_connectivity)