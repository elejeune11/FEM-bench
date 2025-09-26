def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 2)
    expected_nodes = np.array([0.0, 0.5, 1.0])
    expected_elements = np.array([[0, 1], [1, 2]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (3,)
    assert element_connectivity.shape == (2, 2)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 1)
    expected_nodes = np.array([0.0, 1.0])
    expected_elements = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)