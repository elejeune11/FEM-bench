def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    invalid_inputs = [[0.1, 0.2], 42, np.array([0.1, 0.2, 0.3]), np.array([[0.1], [0.2]]), np.array([[0.1, 0.2, 0.3]]), np.array([np.nan, 0.2]), np.array([0.1, np.inf]), np.array([[0.1, 0.2], [0.3, np.nan]]), np.array([]), np.array([[0.1, 0.2], [0.3, 0.4, 0.5]], dtype=object)]
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
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.6, 0.3], [0.1, 0.8], [0.25, 0.25]])
    (N, _) = fcn(pts)
    s = N.sum(axis=1)[:, 0]
    assert np.allclose(s, np.ones(pts.shape[0]), rtol=1e-13, atol=1e-13)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.35, 0.15], [0.2, 0.3], [0.6, 0.3], [0.1, 0.8], [0.25, 0.25]])
    (_, dN) = fcn(pts)
    grad_sum = dN.sum(axis=1)
    zeros = np.zeros_like(grad_sum)
    assert np.allclose(grad_sum, zeros, rtol=1e-13, atol=1e-13)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    A = N[:, :, 0].T
    I = np.eye(6)
    assert np.allclose(A, I, rtol=1e-13, atol=1e-13)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.2, 0.3], [0.6, 0.2], [0.25, 0.25], [1.0 / 3.0, 1.0 / 3.0]])
    coeff_sets = [np.array([0.5, -1.2, 2.3, 0.0, 0.0, 0.0]), np.array([-0.7, 1.1, 0.9, 0.3, -0.4, 2.0])]

    def eval_poly(x, c):
        xi = x[:, 0]
        eta = x[:, 1]
        return c[0] + c[1] * xi + c[2] * eta + c[3] * xi * xi + c[4] * xi * eta + c[5] * eta * eta
    for c in coeff_sets:
        u_nodes = eval_poly(nodes, c)
        (N, _) = fcn(pts)
        u_interp = (N[:, :, 0] * u_nodes[None, :]).sum(axis=1)
        u_true = eval_poly(pts, c)
        assert np.allclose(u_interp, u_true, rtol=1e-13, atol=1e-13)

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.2, 0.3], [0.6, 0.2], [0.25, 0.25], [1.0 / 3.0, 1.0 / 3.0]])
    coeff_sets = [np.array([0.5, -1.2, 2.3, 0.0, 0.0, 0.0]), np.array([-0.7, 1.1, 0.9, 0.3, -0.4, 2.0])]

    def grad_poly(x, c):
        xi = x[:, 0]
        eta = x[:, 1]
        dp_dxi = c[1] + 2.0 * c[3] * xi + c[4] * eta
        dp_deta = c[2] + c[4] * xi + 2.0 * c[5] * eta
        return np.column_stack([dp_dxi, dp_deta])
    for c in coeff_sets:
        u_nodes = c[0] + c[1] * nodes[:, 0] + c[2] * nodes[:, 1] + c[3] * nodes[:, 0] * nodes[:, 0] + c[4] * nodes[:, 0] * nodes[:, 1] + c[5] * nodes[:, 1] * nodes[:, 1]
        (_, dN) = fcn(pts)
        grads_interp = (dN * u_nodes[None, :, None]).sum(axis=1)
        grads_true = grad_poly(pts, c)
        assert np.allclose(grads_interp, grads_true, rtol=1e-12, atol=1e-12)