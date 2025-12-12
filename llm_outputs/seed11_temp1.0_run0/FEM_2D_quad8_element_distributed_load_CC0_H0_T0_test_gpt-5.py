def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test straight edges on a uniformly 2x-scaled Q8 element.
    For each face, apply a constant traction and verify:
    1) The summed nodal forces on the loaded edge equal traction times the physical edge length.
    2) All nodes not on the loaded edge have zero load.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float) * 2.0
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    traction = np.array([1.234, -0.567], dtype=float)
    num_gauss_pts = 2
    for (face, nodes) in face_nodes.items():
        r = fcn(face, node_coords, traction, num_gauss_pts)
        assert isinstance(r, np.ndarray) and r.shape == (16,)
        r_by_node = r.reshape(8, 2)
        edge_force = r_by_node[list(nodes)].sum(axis=0)
        (start, mid, end) = nodes
        p_start = node_coords[start]
        p_end = node_coords[end]
        edge_len = np.linalg.norm(p_end - p_start)
        assert np.allclose(edge_force, traction * edge_len, rtol=1e-12, atol=1e-12)
        all_nodes = set(range(8))
        non_edge_nodes = sorted(all_nodes - set(nodes))
        assert np.allclose(r_by_node[non_edge_nodes], 0.0, atol=1e-12)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test a curved bottom edge (parabolic) on a Q8 element under constant traction.
    The exact total force equals traction times the exact arc length:
    L_exact = sqrt(1 + α) + asinh(sqrt(α)) / sqrt(α), with α = 4 k^2.
    Use 3-point Gauss–Legendre integration and verify total force within a reasonable tolerance.
    """
    c = 0.8
    k = 0.5
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, c + k + 2.0], [-1.0, c + k + 2.0], [0.0, c], [1.0, c + k + 1.0], [0.0, c + k + 2.0], [-1.0, c + k + 1.0]], dtype=float)
    traction = np.array([0.8, 1.1], dtype=float)
    num_gauss_pts = 3
    r = fcn(0, node_coords, traction, num_gauss_pts)
    assert isinstance(r, np.ndarray) and r.shape == (16,)
    r_by_node = r.reshape(8, 2)
    edge_force = r_by_node[[0, 4, 1]].sum(axis=0)
    alpha = 4.0 * k ** 2
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    assert np.allclose(edge_force, traction * L_exact, rtol=0.001, atol=1e-12)