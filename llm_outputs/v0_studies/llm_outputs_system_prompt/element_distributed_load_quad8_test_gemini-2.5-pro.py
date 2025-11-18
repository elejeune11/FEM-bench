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
    traction = np.array([1.5, -2.5])
    num_gauss_pts = 2
    edge_length = 4.0
    expected_total_force = traction * edge_length
    face_nodes_map = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    all_node_indices = set(range(8))
    for (face, loaded_node_indices) in face_nodes_map.items():
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        actual_total_force = np.zeros(2)
        for node_idx in loaded_node_indices:
            actual_total_force += r_elem[2 * node_idx:2 * node_idx + 2]
        assert np.allclose(actual_total_force, expected_total_force)
        unloaded_node_indices = all_node_indices - set(loaded_node_indices)
        for node_idx in unloaded_node_indices:
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
    node_coords = np.zeros((8, 2))
    node_coords[0, :] = [-1.0, c + k]
    node_coords[1, :] = [1.0, c + k]
    node_coords[4, :] = [0.0, c]
    node_coords[3, :] = [-1.0, 3.0]
    node_coords[2, :] = [1.0, 3.0]
    node_coords[7, :] = [-1.0, (node_coords[0, 1] + node_coords[3, 1]) / 2.0]
    node_coords[5, :] = [1.0, (node_coords[1, 1] + node_coords[2, 1]) / 2.0]
    node_coords[6, :] = [0.0, (node_coords[2, 1] + node_coords[3, 1]) / 2.0]
    traction = np.array([10.0, -5.0])
    num_gauss_pts = 3
    face = 0
    alpha = 4.0 * k ** 2
    l_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_total_force = traction * l_exact
    r_elem = fcn(face, node_coords, traction, num_gauss_pts)
    loaded_node_indices = [0, 4, 1]
    actual_total_force = np.zeros(2)
    for node_idx in loaded_node_indices:
        actual_total_force += r_elem[2 * node_idx:2 * node_idx + 2]
    assert np.allclose(actual_total_force, expected_total_force, rtol=0.0001)