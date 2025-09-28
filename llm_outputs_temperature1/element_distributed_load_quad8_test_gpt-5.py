def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straight edge elements scaled by 2x.
    Construct a Q8 element whose edges are straight lines by uniformly scaling
    the reference square by a factor of 2. For each face (0=bottom, 1=right,
    2=top, 3=left), apply a constant traction t = [t_x, t_y]. Verify:
    1) The sum of nodal forces along the loaded edge equals traction * edge length.
    2) All nodes not on the loaded edge receive zero load.
    """
    node_coords = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [0.0, -2.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=float)
    face_nodes = [(0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0)]
    traction = np.array([3.7, -1.9], dtype=float)
    ngp = 2
    tol_zero = 1e-12
    tol_sum = 1e-12
    for (face, edge) in enumerate(face_nodes):
        r = np.ravel(fcn(face, node_coords, traction, ngp))
        L = np.linalg.norm(node_coords[edge[2]] - node_coords[edge[0]])
        sum_fx = sum((r[2 * i] for i in edge))
        sum_fy = sum((r[2 * i + 1] for i in edge))
        summed = np.array([sum_fx, sum_fy])
        expected = traction * L
        assert np.allclose(summed, expected, rtol=0.0, atol=tol_sum)
        other_nodes = set(range(8)) - set(edge)
        for i in other_nodes:
            assert abs(r[2 * i]) <= tol_zero
            assert abs(r[2 * i + 1]) <= tol_zero

def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test constant traction on a curved (parabolic) bottom edge (face=0).
    Bottom edge parameterization: x(s)=s, y(s)=c + k s^2, s ∈ [-1,1],
    realized by setting edge nodes:
        start = (-1, c+k), mid = (0, c), end = (1, c+k).
    With constant traction t = [t_x, t_y], verify that the total nodal force
    on the loaded edge equals t * L_exact, where
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α), α = 4 k^2.
    Use 3-point Gauss–Legendre along the curved edge and a relaxed rtol since
    the integrand is non-polynomial.
    """
    c = 0.3
    k = 1.0
    H = 2.0
    node_coords = np.array([[-1.0, c + k], [1.0, c + k], [1.0, c + k + H], [-1.0, c + k + H], [0.0, c], [1.0, c + k + 0.5 * H], [0.0, c + k + H], [-1.0, c + k + 0.5 * H]], dtype=float)
    traction = np.array([1.25, -0.75], dtype=float)
    ngp = 3
    r = np.ravel(fcn(0, node_coords, traction, ngp))
    edge = (0, 4, 1)
    sum_fx = sum((r[2 * i] for i in edge))
    sum_fy = sum((r[2 * i + 1] for i in edge))
    summed = np.array([sum_fx, sum_fy])
    alpha = 4.0 * k * k
    L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)
    expected = traction * L_exact
    assert np.allclose(summed, expected, rtol=0.01, atol=1e-12)