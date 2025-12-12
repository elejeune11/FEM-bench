def test_basic_mesh_creation(fcn):
    """
    Test basic 1D uniform mesh creation for correctness.
    This test verifies that the provided mesh-generation function produces
    the expected node coordinates and element connectivity for a simple
    1D domain with uniform spacing.
    """
    (x_min, x_max, num_elements) = (0.0, 1.0, 4)
    (nodes, conn) = fcn(x_min, x_max, num_elements)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(conn, np.ndarray)
    assert nodes.ndim == 1
    assert conn.ndim == 2
    num_nodes = num_elements + 1
    assert nodes.shape == (num_nodes,)
    assert conn.shape == (num_elements, 2)
    expected_nodes = np.linspace(x_min, x_max, num_nodes)
    assert np.allclose(nodes, expected_nodes)
    expected_conn = np.array([[0, 1], [1, 2], [2, 3], [3, 4]])
    assert np.array_equal(conn, expected_conn)
    assert np.all(np.diff(nodes) > 0)
    spacing = (x_max - x_min) / num_elements
    assert np.allclose(np.diff(nodes), spacing)
    assert conn.min() == 0
    assert conn.max() == num_nodes - 1
    assert np.all(conn[:, 1] - conn[:, 0] == 1)

def test_single_element_mesh(fcn):
    """
    Test mesh generation for the edge case of a single 1D element.
    This test checks that the mesh-generation function correctly handles
    the minimal valid case of one linear element spanning a domain from
    x_min to x_max. It ensures the function properly computes both node
    coordinates and connectivity for this degenerate case.
    """
    (x_min, x_max, num_elements) = (-2.0, 3.0, 1)
    (nodes, conn) = fcn(x_min, x_max, num_elements)
    assert isinstance(nodes, np.ndarray)
    assert isinstance(conn, np.ndarray)
    assert nodes.shape == (2,)
    assert conn.shape == (1, 2)
    expected_nodes = np.array([x_min, x_max], dtype=float)
    assert np.allclose(nodes, expected_nodes)
    expected_conn = np.array([[0, 1]])
    assert np.array_equal(conn, expected_conn)
    assert nodes[1] > nodes[0]
    assert np.isclose(nodes[1] - nodes[0], x_max - x_min)
    assert conn.min() == 0
    assert conn.max() == 1