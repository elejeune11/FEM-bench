def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min = 0.0
    x_max = 10.0
    num_elements = 4
    num_nodes = num_elements + 1
    expected_nodes = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
    expected_conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray), 'Node coordinates should be a numpy array'
    assert isinstance(element_connectivity, np.ndarray), 'Element connectivity should be a numpy array'
    assert node_coords.shape == (num_nodes,), f'Expected node coordinates shape {(num_nodes,)}, but got {node_coords.shape}'
    assert element_connectivity.shape == (num_elements, 2), f'Expected connectivity shape {(num_elements, 2)}, but got {element_connectivity.shape}'
    assert np.issubdtype(node_coords.dtype, np.floating), 'Node coordinates should be a float type'
    assert np.issubdtype(element_connectivity.dtype, np.integer), 'Element connectivity should be an integer type'
    np.testing.assert_allclose(node_coords, expected_nodes, err_msg='Node coordinates do not match expected values')
    np.testing.assert_array_equal(element_connectivity, expected_conn, err_msg='Element connectivity does not match expected values')

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = -1.0
    x_max = 1.0
    num_elements = 1
    num_nodes = num_elements + 1
    expected_nodes = np.array([-1.0, 1.0])
    expected_conn = np.array([[0, 1]])
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray), 'Node coordinates should be a numpy array'
    assert isinstance(element_connectivity, np.ndarray), 'Element connectivity should be a numpy array'
    assert node_coords.shape == (num_nodes,), f'Expected node coordinates shape {(num_nodes,)}, but got {node_coords.shape}'
    assert element_connectivity.shape == (num_elements, 2), f'Expected connectivity shape {(num_elements, 2)}, but got {element_connectivity.shape}'
    assert np.issubdtype(node_coords.dtype, np.floating), 'Node coordinates should be a float type'
    assert np.issubdtype(element_connectivity.dtype, np.integer), 'Element connectivity should be an integer type'
    np.testing.assert_allclose(node_coords, expected_nodes, err_msg='Node coordinates do not match expected values for a single element')
    np.testing.assert_array_equal(element_connectivity, expected_conn, err_msg='Element connectivity does not match expected values for a single element')