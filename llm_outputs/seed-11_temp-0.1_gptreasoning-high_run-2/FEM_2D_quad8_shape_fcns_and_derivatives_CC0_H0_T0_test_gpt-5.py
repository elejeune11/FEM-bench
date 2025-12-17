def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_types = [None, 42, 'not-array', [0.0, 0.0]]
    for bad in bad_types:
        with pytest.raises(ValueError):
            fcn(bad)
    wrong_shapes = [np.array(1.23), np.zeros((3,)), np.zeros((2, 3)), np.zeros((3, 1)), np.zeros((1, 3))]
    for bad in wrong_shapes:
        with pytest.raises(ValueError):
            fcn(bad)
    non_finite = [np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([[0.0, np.nan], [1.0, 2.0]]), np.array([[0.0, 0.0], [np.inf, 2.0]])]
    for bad in non_finite:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """
    Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    N, _ = fcn(pts)
    sums = N.sum(axis=1).squeeze(-1)
    assert sums.shape == (pts.shape[0],)
    assert np.allclose(sums, np.ones(pts.shape[0]), atol=1e-14, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    _, dN = fcn(pts)
    grad_sums = dN.sum(axis=1)
    assert grad_sums.shape == (pts.shape[0], 2)
    assert np.allclose(grad_sums, np.zeros_like(grad_sums), atol=1e-14, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    N, _ = fcn(nodes)
    M = N[:, :, 0].T
    assert M.shape == (8, 8)
    assert np.allclose(M, np.eye(8), atol=1e-14, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xs = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    ys = np.array([-1.0, -0.25, 0.0, 0.5, 1.0])
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    def poly(vals, P):
        xi = P[:, 0]
        eta = P[:, 1]
        a0, a1, a2, a3, a4, a5 = vals
        return a0 + a1 * xi + a2 * eta + a3 * xi * eta + a4 * xi ** 2 + a5 * eta ** 2
    coefs_linear = np.array([1.23, -0.75, 0.58, 0.0, 0.0, 0.0])
    coefs_quad = np.array([0.12, -0.97, 1.21, -0.33, 0.45, -0.17])
    for coefs in (coefs_linear, coefs_quad):
        nodal_vals = poly(coefs, nodes)
        N, _ = fcn(pts)
        interp = (N[:, :, 0] * nodal_vals).sum(axis=1)
        exact = poly(coefs, pts)
        err = np.max(np.abs(interp - exact))
        assert err < 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xs = np.array([-1.0, -0.6, -0.2, 0.4, 0.9])
    ys = np.array([-1.0, -0.3, 0.0, 0.7, 1.0])
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    def poly(vals, P):
        xi = P[:, 0]
        eta = P[:, 1]
        a0, a1, a2, a3, a4, a5 = vals
        return a0 + a1 * xi + a2 * eta + a3 * xi * eta + a4 * xi ** 2 + a5 * eta ** 2

    def grad_poly(vals, P):
        xi = P[:, 0]
        eta = P[:, 1]
        _, a1, a2, a3, a4, a5 = vals
        dxi = a1 + a3 * eta + 2.0 * a4 * xi
        deta = a2 + a3 * xi + 2.0 * a5 * eta
        return np.column_stack([dxi, deta])
    coefs_linear = np.array([0.77, -0.35, 1.18, 0.0, 0.0, 0.0])
    coefs_quad = np.array([-0.42, 0.91, -1.07, 0.63, -0.28, 0.39])
    for coefs in (coefs_linear, coefs_quad):
        nodal_vals = poly(coefs, nodes)
        _, dN = fcn(pts)
        grad_interp = np.tensordot(dN, nodal_vals, axes=([1], [0]))
        grad_exact = grad_poly(coefs, pts)
        err = np.max(np.abs(grad_interp - grad_exact))
        assert err < 1e-12