def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 4)
    assert len(node_coords) == 5
    assert node_coords[0] == 0.0
    assert node_coords[-1] == 1.0
    assert len(element_connectivity) == 4
    assert element_connectivity.shape == (4, 2)
    assert all((element_connectivity[i][1] == element_connectivity[i + 1][0] for i in range(len(element_connectivity) - 1)))
    assert all((0 <= idx < len(node_coords) for row in element_connectivity for idx in row))

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (node_coords, element_connectivity) = fcn(-1.0, 1.0, 1)
    assert len(node_coords) == 2
    assert node_coords[0] == -1.0
    assert node_coords[-1] == 1.0
    assert len(element_connectivity) == 1
    assert element_connectivity.shape == (1, 2)
    assert element_connectivity[0][0] == 0
    assert element_connectivity[0][1] == 1