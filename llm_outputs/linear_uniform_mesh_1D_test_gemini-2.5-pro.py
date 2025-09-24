def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 10.0
    num_elements = 5
    num_nodes = num_elements + 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_coords = np.linspace(x_min, x_max, num_nodes)
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (num_nodes,)
    assert element_connectivity.shape == (num_elements, 2)
    assert_allclose(node_coords, expected_coords)
    assert_array_equal(element_connectivity, expected_connectivity)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = -5.0
    x_max = 5.0
    num_elements = 1
    num_nodes = num_elements + 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_coords = np.array([-5.0, 5.0])
    expected_connectivity = np.array([[0, 1]])
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (num_nodes,)
    assert element_connectivity.shape == (num_elements, 2)
    assert_allclose(node_coords, expected_coords)
    assert_array_equal(element_connectivity, expected_connectivity)