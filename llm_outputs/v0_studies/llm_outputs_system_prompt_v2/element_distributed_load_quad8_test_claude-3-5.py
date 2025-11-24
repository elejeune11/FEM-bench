def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test traction integral on straight edges with 2x uniform scaling."""
    node_coords = 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([1.2, -0.7])
    edge_length = 4.0
    expected_force = traction * edge_length
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        if face == 0:
            edge_nodes = [0, 4, 1]
        elif face == 1:
            edge_nodes = [1, 5, 2]
        elif face == 2:
            edge_nodes = [2, 6, 3]
        else:
            edge_nodes = [3, 7, 0]
        edge_force = np.zeros(2)
        for n in edge_nodes:
            edge_force += r[2 * n:2 * n + 2]
        assert_allclose(edge_force, expected_force, rtol=1e-14)
        non_edge_nodes = list(set(range(8)) - set(edge_nodes))
        for n in non_edge_nodes:
            assert_allclose(r[2 * n:2 * n + 2], 0, atol=1e-14)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test traction integral on curved parabolic edge."""
    c = 0.5
    k = 0.25
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 1], [-1, 1], [0, c], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([0.8, -1.3])
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_force = traction * L_exact
    r = fcn(0, node_coords, traction, num_gauss_pts=3)
    edge_force = np.zeros(2)
    for n in [0, 4, 1]:
        edge_force += r[2 * n:2 * n + 2]
    assert_allclose(edge_force, expected_force, rtol=1e-06)
    for n in [2, 3, 5, 6, 7]:
        assert_allclose(r[2 * n:2 * n + 2], 0, atol=1e-14)