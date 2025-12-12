def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min = 0.0
    x_max = 10.0
    num_elements = 5
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert node_coords.shape == (num_elements + 1,), f'Expected node_coords shape ({num_elements + 1},), got {node_coords.shape}'
    assert element_connectivity.shape == (num_elements, 2), f'Expected element_connectivity shape ({num_elements}, 2), got {element_connectivity.shape}'
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    np.testing.assert_array_almost_equal(node_coords, expected_coords, decimal=10)
    for i in range(num_elements):
        assert element_connectivity[i, 0] == i, f'Element {i} first node should be {i}, got {element_connectivity[i, 0]}'
        assert element_connectivity[i, 1] == i + 1, f'Element {i} second node should be {i + 1}, got {element_connectivity[i, 1]}'
    assert np.all(np.diff(node_coords) > 0), 'Node coordinates should be monotonically increasing'

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = 0.0
    x_max = 5.0
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert node_coords.shape == (2,), f'Expected node_coords shape (2,), got {node_coords.shape}'
    assert element_connectivity.shape == (1, 2), f'Expected element_connectivity shape (1, 2), got {element_connectivity.shape}'
    expected_coords = np.array([x_min, x_max])
    np.testing.assert_array_almost_equal(node_coords, expected_coords, decimal=10)
    assert element_connectivity[0, 0] == 0, f'Element 0 first node should be 0, got {element_connectivity[0, 0]}'
    assert element_connectivity[0, 1] == 1, f'Element 0 second node should be 1, got {element_connectivity[0, 1]}'