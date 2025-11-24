def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max) = (0.0, 10.0)
    num_elements = 4
    (nodes, connectivity) = fcn(x_min, x_max, num_elements)
    assert len(nodes) == num_elements + 1
    assert nodes[0] == x_min
    assert nodes[-1] == x_max
    assert connectivity.shape == (num_elements, 2)
    assert all(connectivity[:, 1] == connectivity[:, 0] + 1)
    assert all(nodes[connectivity[:, 1]] - nodes[connectivity[:, 0]] == 2.5)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max) = (-1.0, 1.0)
    num_elements = 1
    (nodes, connectivity) = fcn(x_min, x_max, num_elements)
    assert len(nodes) == 2
    assert nodes[0] == x_min
    assert nodes[1] == x_max
    assert connectivity.shape == (1, 2)
    assert connectivity[0, 0] == 0
    assert connectivity[0, 1] == 1