def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    x_min, x_max, num_elements = (0.0, 1.0, 4)
    nodes, conn = fcn(x_min, x_max, num_elements)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(conn, np.ndarray)
    num_nodes_expected = num_elements + 1
    assert nodes.shape == (num_nodes_expected,)
    assert conn.shape == (num_elements, 2)
    assert np.issubdtype(nodes.dtype, np.floating)
    assert np.issubdtype(conn.dtype, np.integer)
    dx = (x_max - x_min) / num_elements
    expected_nodes = x_min + dx * np.arange(num_nodes_expected)
    assert np.allclose(nodes, expected_nodes)
    assert np.all(np.diff(nodes) > 0.0)
    assert np.allclose(np.diff(nodes), dx)
    expected_conn = np.column_stack([np.arange(num_elements), np.arange(1, num_elements + 1)])
    assert np.array_equal(conn, expected_conn)
    assert conn.min() == 0
    assert conn.max() == num_nodes_expected - 1

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    x_min, x_max, num_elements = (-2.5, 2.5, 1)
    nodes, conn = fcn(x_min, x_max, num_elements)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(conn, np.ndarray)
    assert nodes.shape == (2,)
    assert conn.shape == (1, 2)
    assert np.issubdtype(nodes.dtype, np.floating)
    assert np.issubdtype(conn.dtype, np.integer)
    expected_nodes = np.array([x_min, x_max], dtype=nodes.dtype)
    assert np.allclose(nodes, expected_nodes)
    assert np.all(np.diff(nodes) > 0.0)
    assert np.allclose(np.diff(nodes), x_max - x_min)
    expected_conn = np.array([[0, 1]], dtype=conn.dtype)
    assert np.array_equal(conn, expected_conn)
    assert conn.min() == 0
    assert conn.max() == 1