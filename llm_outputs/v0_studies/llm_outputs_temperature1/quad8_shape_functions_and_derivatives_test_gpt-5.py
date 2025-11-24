def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_inputs = [None, 3.14, 'not-an-array', [0.0, 0.0], np.array([1.0, 2.0, 3.0]), np.array([[0.0, 1.0, 2.0]]), np.array([[0.0], [1.0]]), np.array([]), np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([0.0, -np.inf]), np.array([[0.0, 0.0], [np.nan, 1.0]])]
    for bad in bad_inputs:
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
    (N, dN) = fcn(pts)
    s = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(s, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    grad_sum = np.sum(dN, axis=1)
    zeros = np.zeros_like(grad_sum)
    assert np.allclose(grad_sum, zeros, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, dN) = fcn(nodes)
    M = np.swapaxes(N[:, :, 0], 0, 1)
    I = np.eye(8)
    assert np.allclose(M, I, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1.0, 1.0, 5)
    (Xi, Eta) = np.meshgrid(grid, grid, indexing='xy')
    samples = np.column_stack([Xi.ravel(), Eta.ravel()])

    def p1(x, y):
        return 3.0 + 2.0 * x - 1.0 * y

    def p2(x, y):
        return 1.0 - 0.5 * x + 0.75 * y + 0.3 * x * y + 0.2 * x * x - 0.1 * y * y
    for p in (p1, p2):
        p_nodes = p(nodes[:, 0], nodes[:, 1])
        (N, _) = fcn(samples)
        p_interp = N[:, :, 0] @ p_nodes
        p_exact = p(samples[:, 0], samples[:, 1])
        err = np.max(np.abs(p_interp - p_exact))
        assert err < 1e-12

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    grid = np.linspace(-1.0, 1.0, 5)
    (Xi, Eta) = np.meshgrid(grid, grid, indexing='xy')
    samples = np.column_stack([Xi.ravel(), Eta.ravel()])

    def p1(x, y):
        return 3.0 + 2.0 * x - 1.0 * y

    def grad_p1(x, y):
        gx = 2.0 + 0.0 * x
        gy = -1.0 + 0.0 * y
        return (gx, gy)

    def p2(x, y):
        return 1.0 - 0.5 * x + 0.75 * y + 0.3 * x * y + 0.2 * x * x - 0.1 * y * y

    def grad_p2(x, y):
        gx = -0.5 + 0.3 * y + 0.4 * x
        gy = 0.75 + 0.3 * x - 0.2 * 2.0 * y / 2.0
        gy = 0.75 + 0.3 * x - 0.2 * 2.0 * y / 2.0
        gy = 0.75 + 0.3 * x - 0.2 * 2.0 * y / 2.0
        gy = 0.75 + 0.3 * x - 0.2 * y
        return (gx, gy)
    for (p, gradp) in ((p1, grad_p1), (p2, grad_p2)):
        p_nodes = p(nodes[:, 0], nodes[:, 1])
        (_, dN) = fcn(samples)
        gx_approx = dN[:, :, 0] @ p_nodes
        gy_approx = dN[:, :, 1] @ p_nodes
        (gx_exact, gy_exact) = gradp(samples[:, 0], samples[:, 1])
        err_x = np.max(np.abs(gx_approx - gx_exact))
        err_y = np.max(np.abs(gy_approx - gy_exact))
        assert err_x < 1e-12
        assert err_y < 1e-12