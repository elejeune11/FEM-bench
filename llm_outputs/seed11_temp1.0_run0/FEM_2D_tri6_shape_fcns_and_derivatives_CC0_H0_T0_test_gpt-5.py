def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    invalid_inputs = [[0.1, 0.2], (0.1, 0.2), 'not an array', np.array(3.14), np.array([0.1]), np.array([0.1, 0.2, 0.3]), np.array([[0.1], [0.2]]), np.array([[0.1, 0.2, 0.3]]), np.array([np.nan, 0.2]), np.array([0.1, np.inf]), np.array([[0.1, 0.2], [0.3, np.nan]]), np.array([[0.1, 0.2], [0.3, np.inf]])]
    for bad in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.25, 0.25], [0.1, 0.7]])
    (N, _) = fcn(pts)
    sums = N.squeeze(-1).sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)
    (N_single, _) = fcn(np.array([0.3, 0.4]))
    sum_single = N_single.squeeze(-1).sum(axis=1)[0]
    assert abs(sum_single - 1.0) < 1e-12

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.25, 0.25], [0.1, 0.7]])
    (_, dN) = fcn(pts)
    grad_sums = dN.sum(axis=1)
    assert np.allclose(grad_sums, np.zeros_like(grad_sums), atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    M = N.squeeze(-1).swapaxes(0, 1)
    I = np.eye(6)
    assert np.allclose(M, I, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.25, 0.25], [0.1, 0.7]])

    def p1(x):
        xi = x[..., 0]
        eta = x[..., 1]
        return 0.7 * xi - 1.1 * eta + 0.3

    def p2(x):
        xi = x[..., 0]
        eta = x[..., 1]
        return 0.2 * xi * xi + 0.5 * eta * eta - 0.3 * xi * eta + 0.7 * xi - 0.2 * eta + 0.9
    (N_pts, _) = fcn(pts)
    N2 = N_pts.squeeze(-1)
    u_nodes_1 = p1(nodes)
    u_interp_1 = N2 @ u_nodes_1
    u_exact_1 = p1(pts)
    err1 = np.max(np.abs(u_interp_1 - u_exact_1))
    assert err1 < 1e-12
    u_nodes_2 = p2(nodes)
    u_interp_2 = N2 @ u_nodes_2
    u_exact_2 = p2(pts)
    err2 = np.max(np.abs(u_interp_2 - u_exact_2))
    assert err2 < 1e-12

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.25, 0.25], [0.1, 0.7]])

    def p1(x):
        xi = x[..., 0]
        eta = x[..., 1]
        return 0.7 * xi - 1.1 * eta + 0.3

    def grad_p1(x):
        return np.column_stack((np.full(x.shape[0], 0.7), np.full(x.shape[0], -1.1)))

    def p2(x):
        xi = x[..., 0]
        eta = x[..., 1]
        return 0.2 * xi * xi + 0.5 * eta * eta - 0.3 * xi * eta + 0.7 * xi - 0.2 * eta + 0.9

    def grad_p2(x):
        xi = x[..., 0]
        eta = x[..., 1]
        dxi = 0.4 * xi - 0.3 * eta + 0.7
        deta = 1.0 * eta - 0.3 * xi - 0.2
        return np.column_stack((dxi, deta))
    u_nodes_1 = p1(nodes)
    (_, dN_pts) = fcn(pts)
    grad_interp_1 = np.einsum('nid,i->nd', dN_pts, u_nodes_1)
    grad_exact_1 = grad_p1(pts)
    err1 = np.max(np.linalg.norm(grad_interp_1 - grad_exact_1, axis=1))
    assert err1 < 1e-12
    u_nodes_2 = p2(nodes)
    (_, dN_pts) = fcn(pts)
    grad_interp_2 = np.einsum('nid,i->nd', dN_pts, u_nodes_2)
    grad_exact_2 = grad_p2(pts)
    err2 = np.max(np.linalg.norm(grad_interp_2 - grad_exact_2, axis=1))
    assert err2 < 1e-12