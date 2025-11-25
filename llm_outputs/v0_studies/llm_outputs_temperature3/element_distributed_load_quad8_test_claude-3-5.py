def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements with 2x scaling."""
    import numpy as np
    node_coords = 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([1.0, 2.0])
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        if face == 0:
            edge_nodes = [0, 4, 1]
            L = 4.0
        elif face == 1:
            edge_nodes = [1, 5, 2]
            L = 4.0
        elif face == 2:
            edge_nodes = [2, 6, 3]
            L = 4.0
        else:
            edge_nodes = [3, 7, 0]
            L = 4.0
        F_total = np.zeros(2)
        for n in edge_nodes:
            F_total += r[2 * n:2 * n + 2]
        assert np.allclose(F_total, traction * L)
        non_edge = list(set(range(8)) - set(edge_nodes))
        for n in non_edge:
            assert np.allclose(r[2 * n:2 * n + 2], 0)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test constant traction integration on a parabolic curved edge."""
    import numpy as np
    c = 1.0
    k = 0.5
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 2], [-1, 2], [0, c], [1, 1.5], [0, 2], [-1, 1.5]])
    traction = np.array([1.0, -2.0])
    r = fcn(face=0, node_coords=node_coords, traction=traction, num_gauss_pts=3)
    F_total = np.zeros(2)
    for n in [0, 4, 1]:
        F_total += r[2 * n:2 * n + 2]
    assert np.allclose(F_total, traction * L_exact, rtol=0.001)