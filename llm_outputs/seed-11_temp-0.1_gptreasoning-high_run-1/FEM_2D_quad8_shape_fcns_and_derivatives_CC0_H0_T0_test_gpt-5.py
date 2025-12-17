def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0.0, 0.0], np.array([0.0, 0.0, 0.0]), np.zeros((2, 1)), np.zeros((2, 3)), np.array([[0.0, 0.0, 0.0]]), np.array([np.nan, 0.0]), np.array([0.0, np.inf]), np.array([[0.0, 0.0], [np.nan, 1.0]]), None, 'not an array', np.array(0.0), np.zeros((1, 2, 1))]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    N, dN = fcn(pts)
    assert N.shape == (pts.shape[0], 8, 1)
    sums = N.sum(axis=1).squeeze(-1)
    assert np.allclose(sums, np.ones(pts.shape[0]), rtol=0.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    N, dN = fcn(pts)
    grad_sums = dN.sum(axis=1)
    assert np.allclose(grad_sums, np.zeros((pts.shape[0], 2)), rtol=0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    N, _ = fcn(nodes)
    M = N[..., 0].T
    assert M.shape == (8, 8)
    assert np.allclose(M, np.eye(8), rtol=0.0, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    grid = np.linspace(-1.0, 1.0, 5)
    xi, eta = np.meshgrid(grid, grid, indexing='xy')
    samples = np.column_stack([xi.ravel(), eta.ravel()])
    N_samples, _ = fcn(samples)
    Nmat = N_samples[..., 0]
    polys = [lambda x: np.ones(x.shape[0]), lambda x: x[:, 0], lambda x: x[:, 1], lambda x: x[:, 0] ** 2, lambda x: x[:, 0] * x[:, 1], lambda x: x[:, 1] ** 2]
    for p in polys:
        nodal_vals = p(nodes)
        interp = Nmat @ nodal_vals
        truth = p(samples)
        err = np.max(np.abs(interp - truth))
        assert err <= 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    grid = np.linspace(-1.0, 1.0, 5)
    xi, eta = np.meshgrid(grid, grid, indexing='xy')
    samples = np.column_stack([xi.ravel(), eta.ravel()])
    _, dN = fcn(samples)

    def p_const(x):
        return np.ones(x.shape[0])

    def g_const(x):
        return np.zeros((x.shape[0], 2))

    def p_x(x):
        return x[:, 0]

    def g_x(x):
        g = np.zeros((x.shape[0], 2))
        g[:, 0] = 1.0
        return g

    def p_y(x):
        return x[:, 1]

    def g_y(x):
        g = np.zeros((x.shape[0], 2))
        g[:, 1] = 1.0
        return g

    def p_x2(x):
        return x[:, 0] ** 2

    def g_x2(x):
        g = np.zeros((x.shape[0], 2))
        g[:, 0] = 2.0 * x[:, 0]
        return g

    def p_xy(x):
        return x[:, 0] * x[:, 1]

    def g_xy(x):
        g = np.zeros((x.shape[0], 2))
        g[:, 0] = x[:, 1]
        g[:, 1] = x[:, 0]
        return g

    def p_y2(x):
        return x[:, 1] ** 2

    def g_y2(x):
        g = np.zeros((x.shape[0], 2))
        g[:, 1] = 2.0 * x[:, 1]
        return g
    cases = [(p_const, g_const), (p_x, g_x), (p_y, g_y), (p_x2, g_x2), (p_xy, g_xy), (p_y2, g_y2)]
    for p, grad in cases:
        nodal_vals = p(nodes)
        grad_interp = (dN * nodal_vals[None, :, None]).sum(axis=1)
        grad_true = grad(samples)
        err = np.max(np.abs(grad_interp - grad_true))
        assert err <= 1e-12