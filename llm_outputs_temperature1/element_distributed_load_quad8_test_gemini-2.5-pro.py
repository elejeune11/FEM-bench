def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load."""
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    scale = 2.0
    node_coords = scale * ref_coords
    traction = np.array([10.0, -20.0])
    num_gauss_pts = 2
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    edge_length = 2.0 * scale
    F_expected = traction * edge_length
    all_node_indices = set(range(8))
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        loaded_indices = face_nodes[face]
        Fx_total = np.sum(r_elem[2 * np.array(loaded_indices)])
        Fy_total = np.sum(r_elem[2 * np.array(loaded_indices) + 1])
        F_total = np.array([Fx_total, Fy_total])
        assert np.allclose(F_total, F_expected)
        unloaded_indices = list(all_node_indices - set(loaded_indices))
        unloaded_dofs_x = 2 * np.array(unloaded_indices)
        unloaded_dofs_y = 2 * np.array(unloaded_indices) + 1
        unloaded_dofs = np.concatenate([unloaded_dofs_x, unloaded_dofs_y])
        assert np.allclose(r_elem[unloaded_dofs], 0.0)

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
    c = 0.0
    k = 1.0
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, 2.0], [-1.0, 2.0], [0.0, c], [1.0, 1.5], [0.0, 2.0], [-1.0, 1.5]])
    alpha = 4 * k ** 2
    L_exact = np.sqrt(1 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    traction = np.array([5.0, 10.0])
    F_expected = traction * L_exact
    face = 0
    num_gauss_pts = 3
    r_elem = fcn(face, node_coords, traction, num_gauss_pts)
    loaded_indices = [0, 4, 1]
    Fx_total = np.sum(r_elem[2 * np.array(loaded_indices)])
    Fy_total = np.sum(r_elem[2 * np.array(loaded_indices) + 1])
    F_total = np.array([Fx_total, Fy_total])
    assert np.allclose(F_total, F_expected, rtol=0.01)