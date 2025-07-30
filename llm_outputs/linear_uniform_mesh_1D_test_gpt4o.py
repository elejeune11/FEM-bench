def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 10.0
    num_elements = 5
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    expected_element_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    assert np.allclose(node_coords, expected_node_coords), 'Node coordinates are incorrect'
    assert np.array_equal(element_connectivity, expected_element_connectivity), 'Element connectivity is incorrect'

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([0.0, 1.0])
    expected_element_connectivity = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_node_coords), 'Node coordinates are incorrect for single element'
    assert np.array_equal(element_connectivity, expected_element_connectivity), 'Element connectivity is incorrect for single element'