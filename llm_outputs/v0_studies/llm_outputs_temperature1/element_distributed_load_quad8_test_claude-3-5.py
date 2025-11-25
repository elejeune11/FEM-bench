def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements with 2x scaling."""
    import numpy as np
    node_coords = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [0, -2], [2, 0], [0, 2], [-2, 0]])
    traction = np.array([3.0, 4.0])
    lengths = [4.0, 4.0, 4.0, 4.0]
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        edge_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}[face]
        total_force = np.zeros(2)
        for n in edge_nodes:
            total_force += r[2 * n:2 * n + 2]
        expected_force = traction * lengths[face]
        assert np.allclose(total_force, expected_force)
        other_nodes = list(set(range(8)) - set(edge_nodes))
        for n in other_nodes:
            assert np.allclose(r[2 * n:2 * n + 2], 0)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test constant traction integration on curved parabolic edge."""
    import numpy as np
    c = 1.0
    k = 0.5
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 3], [-1, 3], [0, c], [1, 2], [0, 3], [-1, 2]])
    traction = np.array([2.0, -1.0])
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    r = fcn(0, node_coords, traction, num_gauss_pts=3)
    edge_nodes = [0, 4, 1]
    total_force = np.zeros(2)
    for n in edge_nodes:
        total_force += r[2 * n:2 * n + 2]
    expected_force = traction * L_exact
    assert np.allclose(total_force, expected_force, rtol=0.01)