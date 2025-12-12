def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load.
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = ref_nodes * 2.0
    traction = np.array([3.0, -2.0], dtype=float)
    face_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for face in (0, 1, 2, 3):
        r = fcn(face, node_coords, traction, 2)
        assert isinstance(r, np.ndarray)
        assert r.shape == (16,)
        r_nodes = r.reshape(8, 2)
        (start_idx, mid_idx, end_idx) = face_nodes[face]
        total_on_edge = r_nodes[start_idx] + r_nodes[mid_idx] + r_nodes[end_idx]
        start = node_coords[start_idx]
        end = node_coords[end_idx]
        edge_length = np.linalg.norm(end - start)
        expected = traction * edge_length
        assert np.allclose(total_on_edge, expected, rtol=1e-12, atol=1e-12)
        for idx in range(8):
            if idx not in face_nodes[face]:
                assert np.allclose(r_nodes[idx], np.zeros(2), atol=1e-12)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test the performance of curved edges.
    Curved bottom edge (face=0) parameterized by s ∈ [-1, 1]:
        x(s) = s,  y(s) = c + k s^2  (parabola through the three edge nodes)
    realized by placing 8 node quadrilateral edge nodes as:
        start = (-1, c+k),  mid = (0, c),  end = (1, c+k).
    With a constant Cauchy traction t = [t_x, t_y], check that the total force equals
        [t_x, t_y] * L_exact,
    where the exact arc length on [-1,1] is
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α),   α = 4 k^2.
    Note that the function integrates with 3-point Gauss–Legendre along the curved edge.
    The integrand involves sqrt(1+α s^2), which is not a polynomial, so the
    3-point rule is not exact. Select an appropriate relative tolerance to address this.
    """
    c = 0.5
    k = 0.3
    start = np.array([-1.0, c + k])
    mid = np.array([0.0, c])
    end = np.array([1.0, c + k])
    nodes = np.zeros((8, 2), dtype=float)
    nodes[0] = start
    nodes[4] = mid
    nodes[1] = end
    nodes[2] = np.array([1.0, 2.0])
    nodes[3] = np.array([-1.0, 2.0])
    nodes[5] = np.array([1.0, 1.0])
    nodes[6] = np.array([0.0, 2.0])
    nodes[7] = np.array([-1.0, 1.0])
    traction = np.array([2.5, -1.2], dtype=float)
    r = fcn(0, nodes, traction, 3)
    assert isinstance(r, np.ndarray)
    assert r.shape == (16,)
    r_nodes = r.reshape(8, 2)
    edge_total = r_nodes[0] + r_nodes[4] + r_nodes[1]
    alpha = 4.0 * k * k
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / (sqrt_alpha if sqrt_alpha != 0.0 else 1.0)
    expected = traction * L_exact
    assert np.allclose(edge_total, expected, rtol=0.0001, atol=1e-08)