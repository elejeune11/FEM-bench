def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that for straight edges uniformly scaled by 2x, the summed nodal forces
    on the loaded edge equal the applied constant traction times the physical
    edge length, and that all non-edge nodes remain unloaded for all faces.
    """
    coords_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_coords = 2.0 * coords_ref
    traction = np.array([2.345, -0.876])
    face_edges = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for num_gauss_pts in (1, 2, 3):
        for face, edge_nodes in face_edges.items():
            r = fcn(face, node_coords, traction, num_gauss_pts)
            assert isinstance(r, np.ndarray) and r.shape == (16,)
            start = node_coords[edge_nodes[0]]
            end = node_coords[edge_nodes[2]]
            L = np.linalg.norm(end - start)
            expected_total = traction * L
            fx_total = sum((r[2 * n] for n in edge_nodes))
            fy_total = sum((r[2 * n + 1] for n in edge_nodes))
            assert np.allclose([fx_total, fy_total], expected_total, rtol=1e-12, atol=1e-12)
            all_nodes = np.arange(8)
            other_nodes = [n for n in all_nodes if n not in edge_nodes]
            other_dofs = []
            for n in other_nodes:
                other_dofs.extend([2 * n, 2 * n + 1])
            assert np.allclose(r[other_dofs], 0.0, atol=1e-12)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test a curved bottom edge: x(s)=s, y(s)=c+k s^2 using Q8 edge nodes (0,4,1).
    With constant traction, the total force should equal traction times the
    exact arc length L_exact = sqrt(1+α) + asinh(sqrt(α))/sqrt(α), α=4 k^2.
    Uses 3-point Gauss–Legendre; allow a small relative tolerance.
    """
    c = 0.8
    k = 0.5
    H = 1.7
    n0 = np.array([-1.0, c + k])
    n4 = np.array([0.0, c])
    n1 = np.array([1.0, c + k])
    y_top = c + k + H
    n3 = np.array([-1.0, y_top])
    n6 = np.array([0.0, y_top])
    n2 = np.array([1.0, y_top])
    n7 = np.array([-1.0, c + k + 0.5 * H])
    n5 = np.array([1.0, c + k + 0.5 * H])
    node_coords = np.vstack([n0, n1, n2, n3, n4, n5, n6, n7])
    traction = np.array([1.2, -0.7])
    alpha = 4.0 * k * k
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    r = fcn(0, node_coords, traction, 3)
    assert isinstance(r, np.ndarray) and r.shape == (16,)
    edge_nodes = (0, 4, 1)
    fx_total = sum((r[2 * n] for n in edge_nodes))
    fy_total = sum((r[2 * n + 1] for n in edge_nodes))
    expected_total = traction * L_exact
    assert np.allclose([fx_total, fy_total], expected_total, rtol=1e-06, atol=1e-12)