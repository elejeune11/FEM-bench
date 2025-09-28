def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 10.0
    num_elements = 4
    num_nodes = num_elements + 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.linspace(x_min, x_max, num_nodes)
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert isinstance(node_coords, np.ndarray), 'Node coordinates should be a numpy array'
    assert node_coords.shape == (num_nodes,), f'Node coordinates shape should be ({num_nodes},)'
    assert np.allclose(node_coords, expected_nodes), 'Node coordinate values are incorrect'
    assert isinstance(element_connectivity, np.ndarray), 'Element connectivity should be a numpy array'
    assert element_connectivity.shape == (num_elements, 2), f'Element connectivity shape should be ({num_elements}, 2)'
    assert np.issubdtype(element_connectivity.dtype, np.integer), 'Element connectivity dtype should be integer'
    assert np.array_equal(element_connectivity, expected_connectivity), 'Element connectivity values are incorrect'

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = -5.0
    x_max = 5.0
    num_elements = 1
    num_nodes = num_elements + 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([-5.0, 5.0])
    expected_connectivity = np.array([[0, 1]])
    assert isinstance(node_coords, np.ndarray), 'Node coordinates should be a numpy array'
    assert node_coords.shape == (num_nodes,), f'Node coordinates shape should be ({num_nodes},)'
    assert np.allclose(node_coords, expected_nodes), 'Node coordinate values are incorrect for a single element'
    assert isinstance(element_connectivity, np.ndarray), 'Element connectivity should be a numpy array'
    assert element_connectivity.shape == (num_elements, 2), f'Element connectivity shape should be ({num_elements}, 2)'
    assert np.issubdtype(element_connectivity.dtype, np.integer), 'Element connectivity dtype should be integer'
    assert np.array_equal(element_connectivity, expected_connectivity), 'Element connectivity values are incorrect for a single element'