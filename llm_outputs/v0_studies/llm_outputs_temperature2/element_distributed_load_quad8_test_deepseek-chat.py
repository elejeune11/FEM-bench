def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load."""
    node_coords = np.array([[-2, -2], [2, -2], [2, 2], [-2, 2], [0, -2], [2, 0], [0, 2], [-2, 0]])
    traction = np.array([3.0, 4.0])
    edge_length = 4.0
    expected_total_force = traction * edge_length
    edge_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts=2)
        loaded_nodes = edge_nodes[face]
        total_force_x = 0.0
        total_force_y = 0.0
        for node_idx in loaded_nodes:
            total_force_x += r_elem[2 * node_idx]
            total_force_y += r_elem[2 * node_idx + 1]
        assert np.allclose([total_force_x, total_force_y], expected_total_force, rtol=1e-10)
        all_nodes = set(range(8))
        non_edge_nodes = all_nodes - set(loaded_nodes)
        for node_idx in non_edge_nodes:
            assert abs(r_elem[2 * node_idx]) < 1e-12
            assert abs(r_elem[2 * node_idx + 1]) < 1e-12

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
    c = 2.0
    k = 0.5
    alpha = 4 * k ** 2
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    start_node = np.array([-1.0, c + k])
    mid_node = np.array([0.0, c])
    end_node = np.array([1.0, c + k])
    node_coords = np.array([start_node, end_node, [1.0, 3.0], [-1.0, 3.0], mid_node, [1.0, 2.5], [0.0, 3.0], [-1.0, 2.5]])
    traction = np.array([2.0, -1.5])
    expected_total_force = traction * L_exact
    r_elem = fcn(0, node_coords, traction, num_gauss_pts=3)
    total_force_x = r_elem[0] + r_elem[8] + r_elem[2]
    total_force_y = r_elem[1] + r_elem[9] + r_elem[3]
    assert np.allclose([total_force_x, total_force_y], expected_total_force, rtol=0.0001)