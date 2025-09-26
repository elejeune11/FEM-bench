def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    import numpy as np
    (node_coords, element_connectivity) = fcn(0.0, 10.0, 5)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (6,)
    assert element_connectivity.shape == (5, 2)
    assert np.allclose(node_coords, np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))
    assert np.array_equal(element_connectivity, np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]))
    (node_coords2, element_connectivity2) = fcn(-5.0, 5.0, 10)
    assert node_coords2.shape == (11,)
    assert element_connectivity2.shape == (10, 2)
    assert np.allclose(node_coords2[0], -5.0)
    assert np.allclose(node_coords2[-1], 5.0)
    assert np.allclose(np.diff(node_coords2), np.ones(10))
    for i in range(10):
        assert element_connectivity2[i, 0] == i
        assert element_connectivity2[i, 1] == i + 1

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    import numpy as np
    (node_coords, element_connectivity) = fcn(0.0, 1.0, 1)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)
    assert np.allclose(node_coords, np.array([0.0, 1.0]))
    assert np.array_equal(element_connectivity, np.array([[0, 1]]))
    (node_coords2, element_connectivity2) = fcn(-10.0, 10.0, 1)
    assert node_coords2.shape == (2,)
    assert element_connectivity2.shape == (1, 2)
    assert np.allclose(node_coords2, np.array([-10.0, 10.0]))
    assert np.array_equal(element_connectivity2, np.array([[0, 1]]))