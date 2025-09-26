def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test traction integral on straight edge elements with 2x scaling."""
    import numpy as np
    node_coords = 2 * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([3.0, -2.0])
    edge_lengths = [4.0, 4.0, 4.0, 4.0]
    for face in range(4):
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        edge_map = {0: [0, 1, 8, 9, 2, 3], 1: [2, 3, 10, 11, 4, 5], 2: [4, 5, 12, 13, 6, 7], 3: [6, 7, 14, 15, 0, 1]}
        edge_dofs = edge_map[face]
        edge_force = np.array([r[edge_dofs[0]] + r[edge_dofs[2]] + r[edge_dofs[4]], r[edge_dofs[1]] + r[edge_dofs[3]] + r[edge_dofs[5]]])
        expected_force = traction * edge_lengths[face]
        assert np.allclose(edge_force, expected_force)
        non_edge_dofs = np.setdiff1d(np.arange(16), edge_dofs)
        assert np.allclose(r[non_edge_dofs], 0.0)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test traction integral on curved parabolic edge."""
    import numpy as np
    c = 0.0
    k = 0.5
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    node_coords = np.array([[-1, c + k], [1, c + k], [1, 1], [-1, 1], [0, c], [1, 0], [0, 1], [-1, 0]])
    traction = np.array([2.0, 1.0])
    face = 0
    r = fcn(face, node_coords, traction, num_gauss_pts=3)
    total_force = np.array([r[0] + r[8] + r[2], r[1] + r[9] + r[3]])
    expected_force = traction * L_exact
    assert np.allclose(total_force, expected_force, rtol=0.001)