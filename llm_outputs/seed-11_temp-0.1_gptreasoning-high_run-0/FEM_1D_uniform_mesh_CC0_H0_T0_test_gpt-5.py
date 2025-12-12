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
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, (tuple, list)) and len(result) == 2
    (node_coords, element_connectivity) = result
    expected_nodes = np.linspace(x_min, x_max, num_elements + 1)
    assert isinstance(node_coords, np.ndarray)
    assert node_coords.shape == (num_elements + 1,)
    assert np.allclose(node_coords, expected_nodes)
    diffs = np.diff(node_coords)
    assert np.all(diffs > 0)
    assert np.allclose(diffs, diffs[0])
    expected_connectivity = np.array([[i, i + 1] for i in range(num_elements)], dtype=int)
    assert isinstance(element_connectivity, np.ndarray)
    assert element_connectivity.shape == (num_elements, 2)
    assert np.issubdtype(element_connectivity.dtype, np.integer)
    assert np.array_equal(element_connectivity, expected_connectivity)

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min = -2.0
    x_max = 2.0
    num_elements = 1
    result = fcn(x_min, x_max, num_elements)
    assert isinstance(result, (tuple, list)) and len(result) == 2
    (node_coords, element_connectivity) = result
    expected_nodes = np.array([x_min, x_max], dtype=float)
    assert isinstance(node_coords, np.ndarray)
    assert node_coords.shape == (2,)
    assert np.allclose(node_coords, expected_nodes)
    diffs = np.diff(node_coords)
    assert diffs.shape == (1,)
    assert np.isclose(diffs[0], x_max - x_min)
    expected_connectivity = np.array([[0, 1]], dtype=int)
    assert isinstance(element_connectivity, np.ndarray)
    assert element_connectivity.shape == (1, 2)
    assert np.issubdtype(element_connectivity.dtype, np.integer)
    assert np.array_equal(element_connectivity, expected_connectivity)