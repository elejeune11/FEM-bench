def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that for straight edges on a uniformly 2x-scaled Q8 element under a constant traction,
    the total nodal force on the loaded edge equals traction times physical edge length,
    and that all non-edge nodes receive zero load.
    """
    node_coords = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float) * 2.0
    traction = np.array([1.3, -0.7], dtype=float)
    faces = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for face, nodes in faces.items():
        r = fcn(face, node_coords, traction, num_gauss_pts=2)
        r_nodes = r.reshape(8, 2)
        total_force_edge = r_nodes[list(nodes)].sum(axis=0)
        start, _, end = nodes
        L = np.linalg.norm(node_coords[end] - node_coords[start])
        expected_total = traction * L
        assert np.allclose(total_force_edge, expected_total, rtol=1e-12, atol=1e-12)
        other_nodes = [i for i in range(8) if i not in nodes]
        assert np.allclose(r_nodes[other_nodes], 0.0, atol=1e-14)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test a curved bottom edge: x(s)=s, y(s)=c+k s^2 using Q8 edge nodes:
        start=(-1, c+k), mid=(0, c), end=(1, c+k).
    Under constant traction, the total nodal force on the edge should be traction times
    the exact arc length:
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α), α=4 k^2.
    Use 3-point Gauss–Legendre along the curved edge and a moderate rtol for comparison.
    """
    c = 0.3
    k = 0.5
    yb = c + k
    ytop = yb + 2.0
    ymid = (yb + ytop) / 2.0
    node_coords = np.array([[-1.0, yb], [1.0, yb], [1.0, ytop], [-1.0, ytop], [0.0, c], [1.0, ymid], [0.0, ytop], [-1.0, ymid]], dtype=float)
    traction = np.array([2.1, -0.7], dtype=float)
    r = fcn(0, node_coords, traction, num_gauss_pts=3)
    r_nodes = r.reshape(8, 2)
    total_force_edge = r_nodes[[0, 4, 1]].sum(axis=0)
    alpha = 4.0 * k * k
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_total = traction * L_exact
    assert np.allclose(total_force_edge, expected_total, rtol=1e-06, atol=1e-12)