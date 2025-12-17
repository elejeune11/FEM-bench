def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0.0, 0.0], 0.0, np.array(0.0), np.zeros((2, 1)), np.zeros((3,)), np.array([np.nan, 0.0]), np.array([[0.0, 0.0], [np.inf, 0.5]]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.0], [0.0]])]
    for xi in bad_inputs:
        with pytest.raises(ValueError):
            fcn(xi)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    N, _ = fcn(samples)
    S = N.sum(axis=1)[:, 0]
    assert np.allclose(S, np.ones_like(S), atol=1e-13, rtol=0.0)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]], dtype=float)
    _, dN = fcn(samples)
    grad_sum = dN.sum(axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), atol=1e-13, rtol=0.0)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    N, _ = fcn(nodes)
    M = N[:, :, 0]
    I = np.eye(8)
    assert np.allclose(M, I, atol=1e-14, rtol=0.0)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.333333, -0.4], [-0.8, 0.25]], dtype=float)

    def poly1(x, y):
        return 0.7 + 0.2 * x - 0.3 * y

    def poly2(x, y):
        g, d, e, b, a, c = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)
        return g + d * x + e * y + b * x * y + a * x ** 2 + c * y ** 2
    u1 = poly1(nodes[:, 0], nodes[:, 1])
    N_s, _ = fcn(samples)
    vals1 = N_s[:, :, 0] @ u1
    truth1 = poly1(samples[:, 0], samples[:, 1])
    err1 = np.max(np.abs(vals1 - truth1))
    assert err1 < 1e-12
    u2 = poly2(nodes[:, 0], nodes[:, 1])
    N_s, _ = fcn(samples)
    vals2 = N_s[:, :, 0] @ u2
    truth2 = poly2(samples[:, 0], samples[:, 1])
    err2 = np.max(np.abs(vals2 - truth2))
    assert err2 < 1e-12

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.333333, -0.4], [-0.8, 0.25]], dtype=float)

    def poly1(x, y):
        return 0.7 + 0.2 * x - 0.3 * y
    grad1_true = np.tile(np.array([0.2, -0.3], dtype=float), (samples.shape[0], 1))
    u1 = poly1(nodes[:, 0], nodes[:, 1])
    _, dN_s = fcn(samples)
    grad1_pred = np.einsum('nij,j->ni', dN_s, u1)
    err1 = np.max(np.abs(grad1_pred - grad1_true))
    assert err1 < 1e-12
    g, d, e, b, a, c = (0.1, 0.2, -0.3, 0.4, 0.5, -0.6)

    def poly2(x, y):
        return g + d * x + e * y + b * x * y + a * x ** 2 + c * y ** 2
    u2 = poly2(nodes[:, 0], nodes[:, 1])
    _, dN_s = fcn(samples)
    grad2_pred = np.einsum('nij,j->ni', dN_s, u2)
    grad2_true = np.column_stack((d + b * samples[:, 1] + 2 * a * samples[:, 0], e + b * samples[:, 0] + 2 * c * samples[:, 1]))
    err2 = np.max(np.abs(grad2_pred - grad2_true))
    assert err2 < 1e-12