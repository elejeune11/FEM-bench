def test_basic_mesh_creation(fcn):
    """Test basic mesh creation with simple parameters."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 2
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 0.5, 1.0])
    expected_elements = np.array([[0, 1], [1, 2]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (3,)
    assert element_connectivity.shape == (2, 2)

def test_single_element_mesh(fcn):
    """Test edge case with only one element."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 1.0])
    expected_elements = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)

def test_negative_domain(fcn):
    """Test mesh creation with negative coordinates."""
    (x_min, x_max) = (-2.0, 2.0)
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected_elements = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)

def test_non_uniform_spacing(fcn):
    """Test mesh creation with non-unit domain length."""
    (x_min, x_max) = (0.0, 3.0)
    num_elements = 3
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 1.0, 2.0, 3.0])
    expected_elements = np.array([[0, 1], [1, 2], [2, 3]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)

def test_node_ordering(fcn):
    """Test that nodes are in increasing order."""
    (x_min, x_max) = (-1.0, 1.0)
    num_elements = 3
    (node_coords, _) = fcn(x_min, x_max, num_elements)
    assert np.all(np.diff(node_coords) > 0)

def test_element_connectivity_continuity(fcn):
    """Test that elements form a continuous chain."""
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    for i in range(num_elements - 1):
        assert element_connectivity[i, 1] == element_connectivity[i + 1, 0]

def test_zero_length_domain(fcn):
    """Test edge case with zero-length domain."""
    (x_min, x_max) = (2.0, 2.0)
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([2.0, 2.0])
    expected_elements = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_nodes)
    assert np.array_equal(element_connectivity, expected_elements)