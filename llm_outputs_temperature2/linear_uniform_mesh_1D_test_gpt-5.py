def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    expected_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=int)
    assert node_coords.ndim == 1
    assert node_coords.shape[0] == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_conn)
    spacing = (x_max - x_min) / num_elements
    assert np.allclose(np.diff(node_coords), spacing)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max, num_elements) = (-3.0, 7.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert node_coords.ndim == 1
    assert node_coords.shape[0] == 2
    assert element_connectivity.shape == (1, 2)
    assert np.allclose(node_coords, np.array([x_min, x_max]))
    assert np.array_equal(element_connectivity, np.array([[0, 1]], dtype=int))