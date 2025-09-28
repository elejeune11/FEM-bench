def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (node_coords, element_connectivity) = fcn(0.0, 10.0, 5)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (6,)
    assert element_connectivity.shape == (5, 2)
    assert np.allclose(node_coords, np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    assert np.array_equal(element_connectivity, expected_connectivity)
    assert node_coords[0] == 0.0
    assert node_coords[-1] == 10.0
    assert np.all(np.diff(node_coords) > 0)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (node_coords, element_connectivity) = fcn(-1.0, 1.0, 1)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)
    assert np.allclose(node_coords, np.array([-1.0, 1.0]))
    assert np.array_equal(element_connectivity, np.array([[0, 1]]))
    assert node_coords[0] == -1.0
    assert node_coords[1] == 1.0