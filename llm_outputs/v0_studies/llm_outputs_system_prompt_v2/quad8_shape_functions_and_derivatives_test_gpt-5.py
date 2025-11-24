def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """
    The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised.
    """
    bad_type_inputs = [None, 123, 'not an array', [0.0, 0.0], (0.0, 0.0), object()]
    for bad in bad_type_inputs:
        with pytest.raises(ValueError):
            fcn(bad)
    bad_shape_inputs = [np.zeros((2, 1)), np.zeros((1, 1)), np.zeros((3,)), np.zeros((2, 3)), np.zeros((3, 1)), np.zeros((3, 3))]
    for bad in bad_shape_inputs:
        with pytest.raises(ValueError):
            fcn(bad)
    non_finite_inputs = [np.array([np.nan, 0.0]), np.array([0.0, np.inf]), np.array([[0.0, 0.0], [np.nan, 1.0]]), np.array([[np.inf, -np.inf]])]
    for bad in non_finite_inputs:
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
    (N, _) = fcn(pts)
    sums = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(sums, np.ones_like(sums), rtol=1e-13, atol=1e-13)

def test_derivative_partition_of_unity_quad8(fcn):
    """
    Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (_, dN) = fcn(pts)
    grad_sums = np.sum(dN, axis=1)
    zeros = np.zeros_like(grad_sums)
    assert np.allclose(grad_sums, zeros, rtol=1e-13, atol=1e-13)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """
    For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    M = N[:, :, 0].T
    I = np.eye(8)
    assert np.allclose(M, I, rtol=1e-13, atol=1e-13)

def test_value_completeness_quad8(fcn):
    """
    Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.array([-0.8, -0.2, 0.0, 0.3, 0.7])
    eta = np.array([-0.9, -0.1, 0.4, 0.8, 1.0])
    (XI, ETA) = np.meshgrid(xi, eta, indexing='ij')
    samples = np.column_stack([XI.ravel(), ETA.ravel()])
    polynomials = [lambda x, y: 2.7, lambda x, y: x - 0.3, lambda x, y: -1.2 * y + 0.5, lambda x, y: 0.7 + 0.2 * x - 0.3 * y, lambda x, y: x * x, lambda x, y: y * y, lambda x, y: x * y, lambda x, y: 0.7 + 0.2 * x - 0.3 * y + 0.5 * x * x - 0.6 * x * y + 0.9 * y * y]
    for g in polynomials:
        nodal_vals = g(nodes[:, 0], nodes[:, 1])
        (N_s, _) = fcn(samples)
        vals_interp = N_s[:, :, 0] @ nodal_vals
        vals_exact = g(samples[:, 0], samples[:, 1])
        assert np.allclose(vals_interp, vals_exact, rtol=1e-12, atol=1e-12)

def test_gradient_completeness_quad8(fcn):
    """
    Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.array([-0.8, -0.2, 0.0, 0.3, 0.7])
    eta = np.array([-0.9, -0.1, 0.4, 0.8, 1.0])
    (XI, ETA) = np.meshgrid(xi, eta, indexing='ij')
    samples = np.column_stack([XI.ravel(), ETA.ravel()])
    funcs = [(lambda x, y: 2.7, lambda x, y: (np.zeros_like(x), np.zeros_like(y))), (lambda x, y: x - 0.3, lambda x, y: (np.ones_like(x), np.zeros_like(y))), (lambda x, y: -1.2 * y + 0.5, lambda x, y: (np.zeros_like(x), -1.2 * np.ones_like(y))), (lambda x, y: 0.7 + 0.2 * x - 0.3 * y, lambda x, y: (0.2 * np.ones_like(x), -0.3 * np.ones_like(y))), (lambda x, y: x * x, lambda x, y: (2.0 * x, np.zeros_like(y))), (lambda x, y: y * y, lambda x, y: (np.zeros_like(x), 2.0 * y)), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: 0.7 + 0.2 * x - 0.3 * y + 0.5 * x * x - 0.6 * x * y + 0.9 * y * y, lambda x, y: (0.2 + x - 0.6 * y, -0.3 - 0.6 * x + 1.8 * y))]
    for (g, grad) in funcs:
        nodal_vals = g(nodes[:, 0], nodes[:, 1])
        (_, dN) = fcn(samples)
        grad_interp = np.einsum('nij,j->ni', dN, nodal_vals)
        (gx, gy) = grad(samples[:, 0], samples[:, 1])
        grads_exact = np.column_stack([gx, gy])
        assert np.allclose(grad_interp, grads_exact, rtol=1e-12, atol=1e-12)