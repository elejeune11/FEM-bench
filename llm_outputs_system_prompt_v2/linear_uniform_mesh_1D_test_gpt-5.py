def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.ndim == 1
    expected_num_nodes = num_elements + 1
    assert node_coords.shape == (expected_num_nodes,)
    assert np.isclose(node_coords[0], x_min)
    assert np.isclose(node_coords[-1], x_max)
    expected_nodes = np.linspace(x_min, x_max, expected_num_nodes)
    assert np.allclose(node_coords, expected_nodes)
    diffs = np.diff(node_coords)
    assert np.allclose(diffs, (x_max - x_min) / num_elements)
    assert np.all(diffs > 0.0)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.issubdtype(element_connectivity.dtype, np.integer)
    assert element_connectivity.min() >= 0
    assert element_connectivity.max() < expected_num_nodes
    expected_conn = np.column_stack((np.arange(num_elements), np.arange(1, num_elements + 1)))
    assert np.array_equal(element_connectivity, expected_conn)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = -2.0
    x_max = 3.0
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.ndim == 1
    assert node_coords.shape == (2,)
    assert np.isclose(node_coords[0], x_min)
    assert np.isclose(node_coords[1], x_max)
    assert element_connectivity.shape == (1, 2)
    assert np.issubdtype(element_connectivity.dtype, np.integer)
    assert np.array_equal(element_connectivity, np.array([[0, 1]]))
    assert element_connectivity.min() >= 0
    assert element_connectivity.max() == 1