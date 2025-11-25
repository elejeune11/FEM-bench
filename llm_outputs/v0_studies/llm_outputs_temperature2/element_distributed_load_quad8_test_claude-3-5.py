def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements with 2x scaling."""
    node_coords = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [0, -2], [2, 0], [0, 2], [-2, 0]])
    traction = np.array([3.0, -2.0])
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        edge_length = 4.0
        total_force = np.zeros(2)
        if face == 0:
            nodes = [0, 4, 1]
        elif face == 1:
            nodes = [1, 5, 2]
        elif face == 2:
            nodes = [2, 6, 3]
        else:
            nodes = [3, 7, 0]
        for n in nodes:
            total_force += r[2 * n:2 * n + 2]
        assert_allclose(total_force, traction * edge_length, rtol=1e-14)
        unloaded = set(range(8)) - set(nodes)
        for n in unloaded:
            assert_allclose(r[2 * n:2 * n + 2], 0, atol=1e-14)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test constant traction integration on curved parabolic edge."""
    c = 1.0
    k = 0.5
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 3], [-1, 3], [0, c], [1, 2], [0, 3], [-1, 2]])
    traction = np.array([2.0, -1.0])
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    r = fcn(0, node_coords, traction, num_gauss_pts=3)
    total_force = np.zeros(2)
    for n in [0, 4, 1]:
        total_force += r[2 * n:2 * n + 2]
    assert_allclose(total_force, traction * L_exact, rtol=0.001)
    for n in [2, 3, 5, 6, 7]:
        assert_allclose(r[2 * n:2 * n + 2], 0, atol=1e-14)