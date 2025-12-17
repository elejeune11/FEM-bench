def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that for straight edges scaled uniformly by 2x, the total force equals traction times the physical edge length for each face, and that only edge nodes get nonzero loads."""
    node_coords = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [0.0, -2.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=float)
    traction = np.array([3.1, -2.3], dtype=float)
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for face in range(4):
        edge = face_nodes[face]
        L = np.linalg.norm(node_coords[edge[2]] - node_coords[edge[0]])
        for ng in (1, 2, 3):
            r_elem = fcn(face, node_coords, traction, ng)
            assert isinstance(r_elem, np.ndarray)
            assert r_elem.shape == (16,)
            Fx_total = sum((r_elem[2 * i] for i in edge))
            Fy_total = sum((r_elem[2 * i + 1] for i in edge))
            assert np.isclose(Fx_total, traction[0] * L, rtol=1e-12, atol=1e-12)
            assert np.isclose(Fy_total, traction[1] * L, rtol=1e-12, atol=1e-12)
            off_edge = set(range(8)) - set(edge)
            for j in off_edge:
                assert abs(r_elem[2 * j]) < 1e-13
                assert abs(r_elem[2 * j + 1]) < 1e-13

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test total force for a constant traction on a curved (parabolic) bottom edge equals traction times the exact arc length, within a reasonable tolerance for 3-pt Gauss."""
    c = 0.7
    k = 0.5
    h = 2.0
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, c + k + h], [-1.0, c + k + h], [0.0, c], [1.0, c + k + h / 2.0], [0.0, c + k + h], [-1.0, c + k + h / 2.0]], dtype=float)
    traction = np.array([1.75, -0.9], dtype=float)
    alpha = 4.0 * k ** 2
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    r_elem = fcn(0, node_coords, traction, 3)
    assert isinstance(r_elem, np.ndarray)
    assert r_elem.shape == (16,)
    edge = (0, 4, 1)
    Fx_total = sum((r_elem[2 * i] for i in edge))
    Fy_total = sum((r_elem[2 * i + 1] for i in edge))
    assert np.isclose(Fx_total, traction[0] * L_exact, rtol=0.001, atol=1e-12)
    assert np.isclose(Fy_total, traction[1] * L_exact, rtol=0.001, atol=1e-12)