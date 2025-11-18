def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load."""
    scale = 2.0
    node_coords_base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    node_coords = scale * node_coords_base
    traction = np.array([3.0, -2.0])
    num_gauss_pts = 2
    face_nodes = [[0, 4, 1], [1, 5, 2], [2, 6, 3], [3, 7, 0]]
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        edge_nodes = face_nodes[face]
        edge_length = 2.0 * scale
        total_force_x = 0.0
        total_force_y = 0.0
        for (i, node_idx) in enumerate(edge_nodes):
            dof_x = 2 * node_idx
            dof_y = 2 * node_idx + 1
            total_force_x += r_elem[dof_x]
            total_force_y += r_elem[dof_y]
            if i == 1:
                assert abs(r_elem[dof_x] - traction[0] * edge_length / 3.0) < 1e-12
                assert abs(r_elem[dof_y] - traction[1] * edge_length / 3.0) < 1e-12
            else:
                assert abs(r_elem[dof_x] - traction[0] * edge_length / 6.0) < 1e-12
                assert abs(r_elem[dof_y] - traction[1] * edge_length / 6.0) < 1e-12
        assert abs(total_force_x - traction[0] * edge_length) < 1e-12
        assert abs(total_force_y - traction[1] * edge_length) < 1e-12
        for node_idx in range(8):
            if node_idx not in edge_nodes:
                dof_x = 2 * node_idx
                dof_y = 2 * node_idx + 1
                assert abs(r_elem[dof_x]) < 1e-12
                assert abs(r_elem[dof_y]) < 1e-12

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
    node_coords = np.array([[-1, c + k], [1, c + k], [1, c + 1], [-1, c + 1], [0, c], [1, c + 0.5], [0, c + 1], [-1, c + 0.5]])
    traction = np.array([1.5, -1.0])
    num_gauss_pts = 3
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_total_force = traction * L_exact
    r_elem = fcn(0, node_coords, traction, num_gauss_pts)
    edge_nodes = [0, 4, 1]
    total_force_x = 0.0
    total_force_y = 0.0
    for node_idx in edge_nodes:
        dof_x = 2 * node_idx
        dof_y = 2 * node_idx + 1
        total_force_x += r_elem[dof_x]
        total_force_y += r_elem[dof_y]
    computed_total_force = np.array([total_force_x, total_force_y])
    rel_tol = 0.0001
    assert np.allclose(computed_total_force, expected_total_force, rtol=rel_tol)
    for node_idx in range(8):
        if node_idx not in edge_nodes:
            dof_x = 2 * node_idx
            dof_y = 2 * node_idx + 1
            assert abs(r_elem[dof_x]) < 1e-12
            assert abs(r_elem[dof_y]) < 1e-12