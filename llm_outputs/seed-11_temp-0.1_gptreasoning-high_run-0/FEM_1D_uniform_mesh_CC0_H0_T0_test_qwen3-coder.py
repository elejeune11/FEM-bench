def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_connectivity = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert np.allclose(node_coords, expected_node_coords)
    assert np.array_equal(element_connectivity, expected_connectivity)
    assert len(node_coords) == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    (x_min, x_max, num_elements) = (-1.0, 1.0, 1)
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    expected_node_coords = np.array([-1.0, 1.0])
    expected_connectivity = np.array([[0, 1]])
    assert np.allclose(node_coords, expected_node_coords)
    assert np.array_equal(element_connectivity, expected_connectivity)
    assert len(node_coords) == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)