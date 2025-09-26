def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 4
    (nodes, connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(connectivity, np.ndarray)
    assert nodes.ndim == 1
    assert connectivity.ndim == 2
    assert connectivity.shape[1] == 2
    assert len(nodes) == num_elements + 1
    assert len(connectivity) == num_elements
    assert np.isclose(nodes[0], x_min)
    assert np.isclose(nodes[-1], x_max)
    assert np.allclose(np.diff(nodes), (x_max - x_min) / num_elements)
    assert np.all(connectivity[:, 1] == connectivity[:, 0] + 1)
    assert np.all(connectivity >= 0)
    assert np.all(connectivity < len(nodes))

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max) = (-1.0, 1.0)
    num_elements = 1
    (nodes, connectivity) = fcn(x_min, x_max, num_elements)
    assert len(nodes) == 2
    assert len(connectivity) == 1
    assert np.isclose(nodes[0], x_min)
    assert np.isclose(nodes[1], x_max)
    assert connectivity.shape == (1, 2)
    assert np.array_equal(connectivity[0], [0, 1])
    assert np.isclose(nodes[1] - nodes[0], x_max - x_min)