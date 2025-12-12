def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    num_nodes = num_elements + 1
    expected_node_coords = np.linspace(x_min, x_max, num_nodes)
    assert np.allclose(node_coords, expected_node_coords)
    expected_element_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert np.all(element_connectivity == expected_element_connectivity)

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    (x_min, x_max) = (0.0, 1.0)
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([0.0, 1.0])
    assert np.allclose(node_coords, expected_node_coords)
    expected_element_connectivity = np.array([[0, 1]])
    assert np.all(element_connectivity == expected_element_connectivity)