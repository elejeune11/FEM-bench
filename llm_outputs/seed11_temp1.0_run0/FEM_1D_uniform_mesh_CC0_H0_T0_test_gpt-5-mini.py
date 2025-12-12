def test_basic_mesh_creation(fcn):
    """Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    import numpy as np
    x_min = 0.0
    x_max = 1.0
    num_elements = 4
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.ndim == 1
    assert element_connectivity.ndim == 2
    assert node_coords.shape[0] == num_elements + 1
    assert element_connectivity.shape == (num_elements, 2)
    expected_nodes = np.linspace(x_min, x_max, num_elements + 1)
    assert np.allclose(node_coords, expected_nodes)
    assert np.all(np.diff(node_coords) > 0)
    expected_conn = np.vstack([np.arange(num_elements), np.arange(1, num_elements + 1)]).T
    assert np.array_equal(element_connectivity, expected_conn)
    assert np.issubdtype(element_connectivity.dtype, np.integer)

def test_single_element_mesh(fcn):
    """Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    import numpy as np
    x_min = -2.5
    x_max = 3.5
    num_elements = 1
    (node_coords, element_connectivity) = fcn(x_min, x_max, num_elements)
    assert isinstance(node_coords, np.ndarray)
    assert isinstance(element_connectivity, np.ndarray)
    assert node_coords.shape == (2,)
    assert element_connectivity.shape == (1, 2)
    expected_nodes = np.array([x_min, x_max], dtype=float)
    assert np.allclose(node_coords, expected_nodes)
    expected_conn = np.array([[0, 1]], dtype=int)
    assert np.array_equal(element_connectivity, expected_conn)
    assert np.issubdtype(element_connectivity.dtype, np.integer)