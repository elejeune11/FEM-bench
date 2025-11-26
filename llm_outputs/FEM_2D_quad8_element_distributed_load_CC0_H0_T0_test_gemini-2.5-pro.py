def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load.
    """
    ref_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    node_coords = 2.0 * ref_coords
    edge_length = 4.0
    traction = np.array([15.0, -25.0])
    face_nodes_map = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    all_node_indices = set(range(8))
    for (face_idx, nodes_on_face) in face_nodes_map.items():
        r_elem = fcn(face=face_idx, node_coords=node_coords, traction=traction, num_gauss_pts=2)
        expected_total_force = traction * edge_length
        actual_total_force = r_elem.reshape(8, 2)[nodes_on_face].sum(axis=0)
        assert np.allclose(actual_total_force, expected_total_force)
        off_face_nodes = list(all_node_indices - set(nodes_on_face))
        if off_face_nodes:
            off_face_dofs = np.array([[2 * i, 2 * i + 1] for i in off_face_nodes]).flatten()
            assert np.all(r_elem[off_face_dofs] == 0.0)

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
    3-point rule is not exact. Select an appropriate relative tolerance to address this.
    """
    c = 1.0
    k = 0.5
    traction = np.array([10.0, 20.0])
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, 3.0], [-1.0, 3.0], [0.0, c], [1.0, (c + k + 3.0) / 2.0], [0.0, 3.0], [-1.0, (c + k + 3.0) / 2.0]])
    alpha = 4.0 * k ** 2
    if alpha == 0:
        L_exact = 2.0
    else:
        L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected_total_force = traction * L_exact
    r_elem = fcn(face=0, node_coords=node_coords, traction=traction, num_gauss_pts=3)
    nodes_on_face = [0, 4, 1]
    actual_total_force = r_elem.reshape(8, 2)[nodes_on_face].sum(axis=0)
    assert actual_total_force == pytest.approx(expected_total_force, rel=0.015)