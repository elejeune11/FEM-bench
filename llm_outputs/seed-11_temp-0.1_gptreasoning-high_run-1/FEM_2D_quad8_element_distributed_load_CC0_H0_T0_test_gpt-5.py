def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Straight-edge Q8 element uniformly scaled by 2x: verify that for each face,
    the sum of nodal forces equals traction times the physical edge length, and
    all non-edge nodes receive zero load.
    """
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_coords = 2.0 * ref_coords
    traction = np.array([1.2345, -0.9876])
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    face_end_corners = {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 0)}
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts=2)
        assert r_elem.shape == (16,)
        Fx_total = sum((r_elem[2 * i] for i in face_nodes[face]))
        Fy_total = sum((r_elem[2 * i + 1] for i in face_nodes[face]))
        i0, i1 = face_end_corners[face]
        L_edge = np.linalg.norm(node_coords[i1] - node_coords[i0])
        expected_total = traction * L_edge
        assert np.isclose(Fx_total, expected_total[0], rtol=1e-12, atol=1e-12)
        assert np.isclose(Fy_total, expected_total[1], rtol=1e-12, atol=1e-12)
        non_edge = set(range(8)) - set(face_nodes[face])
        for i in sorted(non_edge):
            assert abs(r_elem[2 * i]) <= 1e-12
            assert abs(r_elem[2 * i + 1]) <= 1e-12

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Curved bottom edge (face=0) parameterized by x(s)=s, y(s)=c+k*s^2:
    verify that the total nodal force equals traction times the exact arc length.
    Uses 3-point Gauss; allow a modest relative tolerance since the integrand is non-polynomial.
    """
    c = 0.2
    k = 0.5
    node_coords = np.zeros((8, 2), dtype=float)
    node_coords[0] = [-1.0, c + k]
    node_coords[4] = [0.0, c]
    node_coords[1] = [1.0, c + k]
    top_y = c + k + 2.0
    node_coords[2] = [1.0, top_y]
    node_coords[3] = [-1.0, top_y]
    node_coords[5] = [1.0, 0.5 * (c + k + top_y)]
    node_coords[6] = [0.0, top_y]
    node_coords[7] = [-1.0, 0.5 * (c + k + top_y)]
    traction = np.array([2.0, -3.0])
    r_elem = fcn(0, node_coords, traction, num_gauss_pts=3)
    assert r_elem.shape == (16,)
    edge_nodes = [0, 4, 1]
    Fx_total = sum((r_elem[2 * i] for i in edge_nodes))
    Fy_total = sum((r_elem[2 * i + 1] for i in edge_nodes))
    alpha = 4.0 * k * k
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_total = traction * L_exact
    assert np.isclose(Fx_total, expected_total[0], rtol=0.002, atol=1e-12)
    assert np.isclose(Fy_total, expected_total[1], rtol=0.002, atol=1e-12)
    non_edge = set(range(8)) - set(edge_nodes)
    for i in sorted(non_edge):
        assert abs(r_elem[2 * i]) <= 1e-12
        assert abs(r_elem[2 * i + 1]) <= 1e-12