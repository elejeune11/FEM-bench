def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that for straight edges uniformly scaled by 2x, the summed nodal loads
    on a loaded edge equal traction times the physical edge length, and that
    non-edge nodes receive zero load. This is checked for all four faces.
    """
    node_coords_ref = np.array([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    scale = 2.0
    node_coords = scale * node_coords_ref
    traction = np.array([2.3, -1.7], dtype=float)
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for (face, conn) in face_nodes.items():
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        (start, mid, end) = conn
        L_edge = np.linalg.norm(node_coords[end] - node_coords[start])
        F_edge = np.zeros(2)
        for i in conn:
            F_edge += r[2 * i:2 * i + 2]
        assert np.allclose(F_edge, traction * L_edge, rtol=1e-12, atol=1e-12)
        other_nodes = set(range(8)) - set(conn)
        for i in other_nodes:
            assert np.allclose(r[2 * i:2 * i + 2], 0.0, atol=1e-13)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test a curved bottom edge defined by x(s)=s, y(s)=c+k s^2 with s in [-1,1].
    With constant traction, the total nodal force on the edge must equal the
    traction vector times the exact arc length:
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α),  α = 4 k^2.
    Uses 3-point Gauss integration; allow a small relative tolerance.
    """
    c = 0.5
    k = 0.3
    H_top = 2.0
    node_coords = np.zeros((8, 2), dtype=float)
    node_coords[0] = (-1.0, c + k)
    node_coords[4] = (0.0, c)
    node_coords[1] = (1.0, c + k)
    node_coords[2] = (1.0, H_top)
    node_coords[3] = (-1.0, H_top)
    node_coords[5] = (1.0, 0.5 * (c + k + H_top))
    node_coords[6] = (0.0, H_top)
    node_coords[7] = (-1.0, 0.5 * (c + k + H_top))
    traction = np.array([1.1, -0.7], dtype=float)
    r = fcn(face=0, node_coords=node_coords, traction=traction, num_gauss_pts=3)
    edge_nodes = (0, 4, 1)
    F_edge = np.zeros(2)
    for i in edge_nodes:
        F_edge += r[2 * i:2 * i + 2]
    alpha = 4.0 * k * k
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    assert np.allclose(F_edge, traction * L_exact, rtol=1e-05, atol=1e-12)
    other_nodes = set(range(8)) - set(edge_nodes)
    for i in other_nodes:
        assert np.allclose(r[2 * i:2 * i + 2], 0.0, atol=1e-13)