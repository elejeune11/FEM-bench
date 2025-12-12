def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """Test that the traction integral works on straight edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load.
    """
    scale = 2.0
    node_coords = scale * np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    edge_connectivity = {0: (0, 4, 1), 1: (1, 5, 2), 2: (2, 6, 3), 3: (3, 7, 0)}
    traction = np.array([3.0, 5.0])
    num_gauss_pts = 2
    edge_length = 2.0 * scale
    for face in range(4):
        r_elem = fcn(face, node_coords, traction, num_gauss_pts)
        assert r_elem.shape == (16,), f'Expected shape (16,), got {r_elem.shape}'
        edge_nodes = edge_connectivity[face]
        total_fx = 0.0
        total_fy = 0.0
        for node in edge_nodes:
            total_fx += r_elem[2 * node]
            total_fy += r_elem[2 * node + 1]
        expected_fx = traction[0] * edge_length
        expected_fy = traction[1] * edge_length
        assert np.isclose(total_fx, expected_fx, rtol=1e-10), f'Face {face}: total Fx = {total_fx}, expected {expected_fx}'
        assert np.isclose(total_fy, expected_fy, rtol=1e-10), f'Face {face}: total Fy = {total_fy}, expected {expected_fy}'
        for node in range(8):
            if node not in edge_nodes:
                assert np.isclose(r_elem[2 * node], 0.0, atol=1e-14), f'Face {face}: node {node} Fx should be zero, got {r_elem[2 * node]}'
                assert np.isclose(r_elem[2 * node + 1], 0.0, atol=1e-14), f'Face {face}: node {node} Fy should be zero, got {r_elem[2 * node + 1]}'

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
    c = -1.0
    k = 0.5
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, 1.0], [-1.0, 1.0], [0.0, c], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    traction = np.array([2.0, 7.0])
    num_gauss_pts = 3
    face = 0
    alpha = 4.0 * k * k
    sqrt_alpha = np.sqrt(alpha)
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(sqrt_alpha) / sqrt_alpha
    r_elem = fcn(face, node_coords, traction, num_gauss_pts)
    assert r_elem.shape == (16,), f'Expected shape (16,), got {r_elem.shape}'
    edge_nodes = (0, 4, 1)
    total_fx = 0.0
    total_fy = 0.0
    for node in edge_nodes:
        total_fx += r_elem[2 * node]
        total_fy += r_elem[2 * node + 1]
    expected_fx = traction[0] * L_exact
    expected_fy = traction[1] * L_exact
    rtol = 0.001
    assert np.isclose(total_fx, expected_fx, rtol=rtol), f'Total Fx = {total_fx}, expected {expected_fx}'
    assert np.isclose(total_fy, expected_fy, rtol=rtol), f'Total Fy = {total_fy}, expected {expected_fy}'
    for node in range(8):
        if node not in edge_nodes:
            assert np.isclose(r_elem[2 * node], 0.0, atol=1e-14), f'Node {node} Fx should be zero, got {r_elem[2 * node]}'
            assert np.isclose(r_elem[2 * node + 1], 0.0, atol=1e-14), f'Node {node} Fy should be zero, got {r_elem[2 * node + 1]}'