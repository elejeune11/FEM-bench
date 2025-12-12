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
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]) * 2.0
    traction = np.array([3.4, -1.1])
    num_gauss_pts = 2
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    for (face, nodes) in face_nodes.items():
        r = fcn(face, node_coords, traction, num_gauss_pts)
        assert r.shape == (16,)
        (start, _, end) = nodes
        L_edge = np.linalg.norm(node_coords[end] - node_coords[start])
        Fx_sum = sum((r[2 * n] for n in nodes))
        Fy_sum = sum((r[2 * n + 1] for n in nodes))
        assert np.isclose(Fx_sum, traction[0] * L_edge, rtol=1e-12, atol=1e-12)
        assert np.isclose(Fy_sum, traction[1] * L_edge, rtol=1e-12, atol=1e-12)
        other_nodes = [n for n in range(8) if n not in nodes]
        other_dof_indices = np.array([2 * n for n in other_nodes] + [2 * n + 1 for n in other_nodes])
        assert np.allclose(r[other_dof_indices], 0.0, atol=1e-12)

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
    c = -1.0
    k = 0.75
    node_coords = np.zeros((8, 2), dtype=float)
    node_coords[0] = [-1.0, c + k]
    node_coords[1] = [1.0, c + k]
    node_coords[4] = [0.0, c]
    top_y = c + k + 2.0
    node_coords[2] = [1.0, top_y]
    node_coords[3] = [-1.0, top_y]
    node_coords[5] = [1.0, 0.5 * (c + k + top_y)]
    node_coords[6] = [0.0, top_y]
    node_coords[7] = [-1.0, 0.5 * (c + k + top_y)]
    traction = np.array([1.2, -2.3])
    num_gauss_pts = 3
    r = fcn(0, node_coords, traction, num_gauss_pts)
    assert r.shape == (16,)
    alpha = 4.0 * k * k
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    Fx_sum = r[2 * 0] + r[2 * 4] + r[2 * 1]
    Fy_sum = r[2 * 0 + 1] + r[2 * 4 + 1] + r[2 * 1 + 1]
    assert np.isclose(Fx_sum, traction[0] * L_exact, rtol=0.005, atol=1e-12)
    assert np.isclose(Fy_sum, traction[1] * L_exact, rtol=0.005, atol=1e-12)