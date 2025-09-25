def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 2
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([0.0, 0.5, 1.0])
    expected_element_connectivity = np.array([[0, 1], [1, 2]])
    np.testing.assert_array_almost_equal(node_coords, expected_node_coords)
    np.testing.assert_array_equal(element_connectivity, expected_element_connectivity)
    assert node_coords.shape == (num_elements + 1,)
    assert element_connectivity.shape == (num_elements, 2)
    assert node_coords[0] == x_min
    assert node_coords[-1] == x_max

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    x_min = -1.0
    x_max = 1.0
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([-1.0, 1.0])
    expected_element_connectivity = np.array([[0, 1]])
    np.testing.assert_array_almost_equal(node_coords, expected_node_coords)
    np.testing.assert_array_equal(element_connectivity, expected_element_connectivity)
    assert node_coords.shape == (num_elements + 1,)
    assert element_connectivity.shape == (num_elements, 2)
    assert node_coords[0] == x_min
    assert node_coords[-1] == x_max