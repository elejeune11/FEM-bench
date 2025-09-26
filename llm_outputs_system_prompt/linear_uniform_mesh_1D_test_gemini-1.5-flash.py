def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 2)
    assert np.allclose(node_coords, [0.0, 0.5, 1.0])
    assert np.allclose(element_connectivity, [[0, 1], [1, 2]])

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 1)
    assert np.allclose(node_coords, [0.0, 1.0])
    assert np.allclose(element_connectivity, [[0, 1]])