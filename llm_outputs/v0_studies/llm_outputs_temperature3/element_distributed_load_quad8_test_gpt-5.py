def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straight edge elements scaled by 2x.
    For each face, verify:
    1) The sum of nodal loads on the loaded edge equals traction * physical edge length.
    2) All nodes not on the loaded edge have zero load.
    """
    node_coords = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [0.0, -2.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=float)
    traction = np.array([2.5, -3.0], dtype=float)
    num_gauss_pts = 2
    edge_conn = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    edge_endpoints = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts)
        assert r.shape == (16,)
        (i0, i1) = edge_endpoints[face]
        L = np.linalg.norm(node_coords[i1] - node_coords[i0])
        (n_start, n_mid, n_end) = edge_conn[face]
        sum_fx = r[2 * n_start] + r[2 * n_mid] + r[2 * n_end]
        sum_fy = r[2 * n_start + 1] + r[2 * n_mid + 1] + r[2 * n_end + 1]
        sum_edge = np.array([sum_fx, sum_fy])
        expected = traction * L
        assert np.allclose(sum_edge, expected, rtol=1e-12, atol=1e-12)
        non_edge_nodes = [n for n in range(8) if n not in edge_conn[face]]
        non_edge_dofs = []
        for n in non_edge_nodes:
            non_edge_dofs.extend([2 * n, 2 * n + 1])
        assert np.allclose(r[non_edge_dofs], 0.0, atol=1e-13)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test total force on a curved parabolic bottom edge under constant traction.
    Curved bottom edge: x(s)=s, y(s)=c+k s^2 for s in [-1,1], realized via:
        start=(-1, c+k), mid=(0, c), end=(1, c+k).
    Verify that the total force equals traction * L_exact with:
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α), α = 4 k^2.
    Uses 3-point Gauss–Legendre for integration; allow a small relative tolerance.
    """
    c = 0.3
    k = 0.4
    H = 2.0
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, c + k + H], [-1.0, c + k + H], [0.0, c], [1.0, c + k + H / 2.0], [0.0, c + k + H], [-1.0, c + k + H / 2.0]], dtype=float)
    traction = np.array([1.2, -0.7], dtype=float)
    num_gauss_pts = 3
    r = fcn(0, node_coords, traction, num_gauss_pts)
    assert r.shape == (16,)
    edge_nodes = (0, 4, 1)
    sum_fx = r[2 * edge_nodes[0]] + r[2 * edge_nodes[1]] + r[2 * edge_nodes[2]]
    sum_fy = r[2 * edge_nodes[0] + 1] + r[2 * edge_nodes[1] + 1] + r[2 * edge_nodes[2] + 1]
    sum_edge = np.array([sum_fx, sum_fy])
    alpha = 4.0 * k * k
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    expected = traction * L_exact
    assert np.allclose(sum_edge, expected, rtol=0.0001, atol=1e-09)