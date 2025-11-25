def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straight edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load.
    """
    node_coords = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [0, -2], [2, 0], [0, 2], [-2, 0]])
    traction = np.array([1.0, 1.0])
    num_gauss_pts = 2
    edge_lengths = [4, 4, 4, 4]
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        total_force = np.sum(r_elem.reshape(8, 2)[[0, 4, 1] if face == 0 else [1, 5, 2] if face == 1 else [2, 6, 3] if face == 2 else [3, 7, 0]], axis=0)
        expected_force = traction * edge_lengths[face]
        assert np.allclose(total_force, expected_force), f'Total force mismatch on face {face}'
        for i in range(8):
            if i not in ([0, 4, 1] if face == 0 else [1, 5, 2] if face == 1 else [2, 6, 3] if face == 2 else [3, 7, 0]):
                assert np.allclose(r_elem[2 * i:2 * i + 2], 0), f'Non-zero load on node {i} for face {face}'

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
    c = 1.0
    k = 0.5
    node_coords = np.array([[-1, c + k], [1, c + k], [1, c + k], [-1, c + k], [0, c], [1, c + k], [0, c + k], [-1, c + k]])
    traction = np.array([1.0, 1.0])
    num_gauss_pts = 3
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    r_elem = fcn(0, node_coords, traction, num_gauss_pts)
    total_force = np.sum(r_elem.reshape(8, 2)[[0, 4, 1]], axis=0)
    expected_force = traction * L_exact
    assert np.allclose(total_force, expected_force, rtol=0.001), 'Total force mismatch on curved edge'