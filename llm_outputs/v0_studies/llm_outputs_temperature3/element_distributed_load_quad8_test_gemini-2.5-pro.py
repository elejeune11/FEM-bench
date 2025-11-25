def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load."""
    node_coords = 2.0 * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    edge_length = 4.0
    traction = np.array([10.0, -5.0])
    num_gauss_pts = 2
    face_node_indices = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    all_node_indices = set(range(8))
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        on_face_nodes = face_node_indices[face]
        total_force = np.zeros(2)
        for node_idx in on_face_nodes:
            total_force += r_elem[2 * node_idx:2 * node_idx + 2]
        expected_force = traction * edge_length
        assert np.allclose(total_force, expected_force)
        off_face_nodes = all_node_indices - set(on_face_nodes)
        for node_idx in off_face_nodes:
            assert np.allclose(r_elem[2 * node_idx:2 * node_idx + 2], 0.0)

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
    c = 1.0
    k = 0.5
    face = 0
    num_gauss_pts = 3
    traction = np.array([15.0, -25.0])
    node1_coords = np.array([-1.0, c + k])
    node5_coords = np.array([0.0, c])
    node2_coords = np.array([1.0, c + k])
    node4_coords = np.array([-1.0, 3.0])
    node3_coords = np.array([1.0, 3.0])
    node8_coords = 0.5 * (node1_coords + node4_coords)
    node6_coords = 0.5 * (node2_coords + node3_coords)
    node7_coords = 0.5 * (node3_coords + node4_coords)
    node_coords = np.array([node1_coords, node2_coords, node3_coords, node4_coords, node5_coords, node6_coords, node7_coords, node8_coords])
    alpha = 4 * k ** 2
    if np.isclose(alpha, 0):
        L_exact = 2.0
    else:
        sqrt_alpha = np.sqrt(alpha)
        L_exact = np.sqrt(1 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    r_elem = fcn(face, node_coords, traction, num_gauss_pts)
    on_face_nodes = [0, 4, 1]
    total_force = np.zeros(2)
    for node_idx in on_face_nodes:
        total_force += r_elem[2 * node_idx:2 * node_idx + 2]
    expected_force = traction * L_exact
    assert np.allclose(total_force, expected_force, rtol=0.001)