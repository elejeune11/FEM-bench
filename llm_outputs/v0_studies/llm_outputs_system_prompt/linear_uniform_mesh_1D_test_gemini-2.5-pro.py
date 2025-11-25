def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    num_nodes = num_elements + 1
    expected_nodes = np.linspace(x_min, x_max, num_nodes)
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert isinstance(node_coords, np.ndarray)
    assert node_coords.shape == (num_nodes,)
    assert np.allclose(node_coords, expected_nodes)
    assert isinstance(element_connectivity, np.ndarray)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.array_equal(element_connectivity, expected_connectivity)
    assert np.issubdtype(element_connectivity.dtype, np.integer)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max) = (-1.0, 1.0)
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    num_nodes = num_elements + 1
    expected_nodes = np.array([-1.0, 1.0])
    expected_connectivity = np.array([[0, 1]])
    assert node_coords.shape == (num_nodes,)
    assert np.allclose(node_coords, expected_nodes)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.array_equal(element_connectivity, expected_connectivity)
    assert np.issubdtype(element_connectivity.dtype, np.integer)