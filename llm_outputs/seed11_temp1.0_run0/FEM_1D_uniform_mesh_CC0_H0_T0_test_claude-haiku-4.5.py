def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(node_coords, expected_nodes), f'Expected node_coords {expected_nodes}, got {node_coords}'
    assert len(node_coords) == num_elements + 1, f'Expected {num_elements + 1} nodes, got {len(node_coords)}'
    assert element_connectivity.shape == (num_elements, 2), f'Expected connectivity shape ({num_elements}, 2), got {element_connectivity.shape}'
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert np.array_equal(element_connectivity, expected_connectivity), f'Expected connectivity {expected_connectivity}, got {element_connectivity}'
    for (i, elem) in enumerate(element_connectivity):
        assert elem[0] == i and elem[1] == i + 1, f'Element {i} has incorrect node indices: {elem}'

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    (x_min, x_max, num_elements) = (2.0, 5.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_nodes = np.array([2.0, 5.0])
    assert np.allclose(node_coords, expected_nodes), f'Expected node_coords {expected_nodes}, got {node_coords}'
    assert len(node_coords) == 2, f'Expected 2 nodes for single element, got {len(node_coords)}'
    assert element_connectivity.shape == (1, 2), f'Expected connectivity shape (1, 2), got {element_connectivity.shape}'
    expected_connectivity = np.array([[0, 1]])
    assert np.array_equal(element_connectivity, expected_connectivity), f'Expected connectivity {expected_connectivity}, got {element_connectivity}'