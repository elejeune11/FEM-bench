def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements with 2x scaling."""
    node_coords = 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([1.0, 2.0])
    edge_length = 4.0
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
        total_force = np.zeros(2)
        for n in edge_nodes:
            total_force += r[2 * n:2 * n + 2]
        assert_allclose(total_force, traction * edge_length)
        other_nodes = list(set(range(8)) - set(edge_nodes))
        for n in other_nodes:
            assert_allclose(r[2 * n:2 * n + 2], 0)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test constant traction integration on curved parabolic edge."""
    c = 1.0
    k = 0.5
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 2], [-1, 2], [0, c], [1, 1.5], [0, 2], [-1, 1.5]])
    traction = np.array([1.0, 2.0])
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    r = fcn(0, node_coords, traction, num_gauss_pts=3)
    total_force = np.zeros(2)
    for n in [0, 4, 1]:
        total_force += r[2 * n:2 * n + 2]
    assert_allclose(total_force, traction * L_exact, rtol=0.01)