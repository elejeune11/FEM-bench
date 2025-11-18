def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max, num_elements) = (0.0, 1.0, 3)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = 4
    expected_coords = np.array([0.0, 1 / 3, 2 / 3, 1.0])
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3]])
    assert node_coords.shape == (expected_nodes,)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.allclose(node_coords, expected_coords)
    assert np.array_equal(element_connectivity, expected_connectivity)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max, num_elements) = (0.0, 1.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = 2
    expected_coords = np.array([0.0, 1.0])
    expected_connectivity = np.array([[0, 1]])
    assert node_coords.shape == (expected_nodes,)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.allclose(node_coords, expected_coords)
    assert np.array_equal(element_connectivity, expected_connectivity)