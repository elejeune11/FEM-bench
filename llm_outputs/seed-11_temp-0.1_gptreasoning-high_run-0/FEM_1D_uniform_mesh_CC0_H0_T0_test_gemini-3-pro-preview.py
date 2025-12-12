def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min = 0.0
    x_max = 2.0
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert isinstance(node_coords, np.ndarray), 'Node coordinates should be a numpy array.'
    assert isinstance(element_connectivity, np.ndarray), 'Element connectivity should be a numpy array.'
    assert node_coords.shape == (num_elements + 1,), f'Expected node_coords shape {(num_elements + 1,)}, got {node_coords.shape}'
    assert element_connectivity.shape == (num_elements, 2), f'Expected connectivity shape {(num_elements, 2)}, got {element_connectivity.shape}'
    np.testing.assert_allclose(node_coords, expected_coords, err_msg='Node coordinates do not match expected uniform distribution.')
    np.testing.assert_array_equal(element_connectivity, expected_connectivity, err_msg='Element connectivity does not match expected topology.')

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = -1.5
    x_max = 1.5
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_coords = np.array([x_min, x_max])
    expected_connectivity = np.array([[0, 1]])
    assert len(node_coords) == 2, 'Single element mesh must have exactly 2 nodes.'
    assert len(element_connectivity) == 1, 'Single element mesh must have exactly 1 connectivity entry.'
    np.testing.assert_allclose(node_coords, expected_coords, err_msg='Node coordinates for single element incorrect.')
    np.testing.assert_array_equal(element_connectivity, expected_connectivity, err_msg='Connectivity for single element incorrect.')