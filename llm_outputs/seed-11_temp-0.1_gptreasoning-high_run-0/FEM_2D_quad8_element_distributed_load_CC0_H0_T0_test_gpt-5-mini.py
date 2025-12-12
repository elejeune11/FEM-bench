def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load."""
    import numpy as np
    scale = 2.0
    ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = ref * scale
    traction = np.array([3.0, -2.0], dtype=float)
    num_gauss_pts = 2
    face_to_nodes = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    for (face, nodes) in face_to_nodes.items():
        r = fcn(face, node_coords, traction, num_gauss_pts)
        assert isinstance(r, np.ndarray)
        assert r.shape == (16,)
        total = np.zeros(2, dtype=float)
        for nid in nodes:
            total += r[2 * nid:2 * nid + 2]
        expected_length = np.linalg.norm(node_coords[nodes[2]] - node_coords[nodes[0]])
        expected_total = traction * expected_length
        assert np.allclose(total, expected_total, rtol=1e-12, atol=1e-12)
        off_nodes = set(range(8)) - set(nodes)
        for nid in off_nodes:
            comp = r[2 * nid:2 * nid + 2]
            assert np.allclose(comp, 0.0, atol=1e-12)

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """Test the performance of curved edges.
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
    3-point rule is not exact. Select an appropriate relative tolerance to address this."""
    import numpy as np
    c = 0.5
    k = 0.25
    H = 1.0
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, c + k + H], [-1.0, c + k + H], [0.0, c], [1.0, c + k + 0.5 * H], [0.0, c + k + H], [-1.0, c + k + 0.5 * H]], dtype=float)
    traction = np.array([1.7, -2.3], dtype=float)
    num_gauss_pts = 3
    r = fcn(0, node_coords, traction, num_gauss_pts)
    assert isinstance(r, np.ndarray)
    assert r.shape == (16,)
    nodes = (0, 4, 1)
    total = np.zeros(2, dtype=float)
    for nid in nodes:
        total += r[2 * nid:2 * nid + 2]
    alpha = 4.0 * k * k
    if alpha == 0.0:
        L_exact = 2.0
    else:
        sqrt_alpha = np.sqrt(alpha)
        L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    expected_total = traction * L_exact
    assert np.allclose(total, expected_total, rtol=1e-06, atol=1e-12)
    off_nodes = set(range(8)) - set(nodes)
    for nid in off_nodes:
        comp = r[2 * nid:2 * nid + 2]
        assert np.allclose(comp, 0.0, atol=1e-12)