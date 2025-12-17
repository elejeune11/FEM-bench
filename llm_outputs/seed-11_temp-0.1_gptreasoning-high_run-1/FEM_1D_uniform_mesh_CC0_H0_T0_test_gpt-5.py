def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    node_coords = np.asarray(node_coords, dtype=float)
    element_connectivity = np.asarray(element_connectivity, dtype=int)
    expected_num_nodes = num_elements + 1
    assert node_coords.ndim == 1
    assert node_coords.shape[0] == expected_num_nodes
    expected_nodes = np.linspace(x_min, x_max, expected_num_nodes)
    assert np.allclose(node_coords, expected_nodes, rtol=1e-12, atol=1e-12)
    assert element_connectivity.shape == (num_elements, 2)
    expected_connectivity = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])
    assert np.array_equal(element_connectivity, expected_connectivity)
    diffs = np.diff(node_coords)
    assert np.all(diffs > 0)
    expected_spacing = (x_max - x_min) / num_elements
    assert np.allclose(diffs, expected_spacing, rtol=1e-12, atol=1e-12)
    assert element_connectivity.min() == 0
    assert element_connectivity.max() == expected_num_nodes - 1

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = -2.5
    x_max = 3.0
    num_elements = 1
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    node_coords = np.asarray(node_coords, dtype=float)
    element_connectivity = np.asarray(element_connectivity, dtype=int)
    assert node_coords.ndim == 1
    assert node_coords.shape == (2,)
    expected_nodes = np.array([x_min, x_max], dtype=float)
    assert np.allclose(node_coords, expected_nodes, rtol=1e-12, atol=1e-12)
    assert element_connectivity.shape == (1, 2)
    assert np.array_equal(element_connectivity, np.array([[0, 1]], dtype=int))
    diffs = np.diff(node_coords)
    assert np.all(diffs > 0)
    assert np.isclose(diffs[0], x_max - x_min, rtol=1e-12, atol=1e-12)
    assert set(element_connectivity.ravel().tolist()) == {0, 1}