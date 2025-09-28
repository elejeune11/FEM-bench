def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 4)
    assert len(node_coords) == 5
    assert node_coords[0] == 0.0
    assert node_coords[-1] == 1.0
    assert np.allclose(node_coords, np.linspace(0.0, 1.0, 5))
    assert element_connectivity.shape == (4, 2)
    assert np.all(element_connectivity[:, 1] - element_connectivity[:, 0] == 1)
    assert np.all(element_connectivity >= 0)
    assert np.all(element_connectivity < len(node_coords))

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (node_coords, element_connectivity) = fcn(-1.0, 1.0, 1)
    assert len(node_coords) == 2
    assert node_coords[0] == -1.0
    assert node_coords[-1] == 1.0
    assert np.allclose(node_coords, [-1.0, 1.0])
    assert element_connectivity.shape == (1, 2)
    assert np.array_equal(element_connectivity, [[0, 1]])