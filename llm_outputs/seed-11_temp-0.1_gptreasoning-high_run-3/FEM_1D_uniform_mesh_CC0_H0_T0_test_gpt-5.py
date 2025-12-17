def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing."""
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.ndim == 1
    assert node_coords.shape[0] == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)
    expected_coords = np.linspace(x_min, x_max, num_elements + 1)
    np.testing.assert_allclose(node_coords, expected_coords, rtol=0, atol=1e-12)
    dx = (x_max - x_min) / num_elements
    diffs = np.diff(node_coords)
    np.testing.assert_allclose(diffs, np.full(num_elements, dx), rtol=0, atol=1e-12)
    assert np.all(diffs > 0)
    expected_connectivity = np.column_stack((np.arange(num_elements), np.arange(1, num_elements + 1)))
    np.testing.assert_array_equal(element_connectivity, expected_connectivity)
    assert element_connectivity.min() >= 0
    assert element_connectivity.max() == num_elements
    assert np.all(element_connectivity[:, 1] - element_connectivity[:, 0] == 1)

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case."""
    x_min = -2.5
    x_max = 3.5
    num_elements = 1
    node_coords, element_connectivity = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.ndim == 1
    assert node_coords.shape[0] == 2
    assert element_connectivity.shape == (1, 2)
    expected_coords = np.array([x_min, x_max], dtype=float)
    np.testing.assert_allclose(node_coords, expected_coords, rtol=0, atol=1e-12)
    diffs = np.diff(node_coords)
    np.testing.assert_allclose(diffs, np.array([x_max - x_min]), rtol=0, atol=1e-12)
    assert diffs[0] > 0
    expected_connectivity = np.array([[0, 1]])
    np.testing.assert_array_equal(element_connectivity, expected_connectivity)
    assert element_connectivity.min() == 0
    assert element_connectivity.max() == 1