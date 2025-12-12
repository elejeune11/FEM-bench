def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError. This test feeds a set of
    bad inputs (wrong type, wrong shape, non-finite) and asserts ValueError is raised."""
    bad_inputs = [[0.0, 0.0], 123, 'invalid', np.array([0.0, 0.0, 0.0]), np.array([[0.0, 0.0, 0.0]]), np.zeros((2, 1)), np.zeros((1, 2, 1)), np.array([np.nan, 0.0]), np.array([np.inf, 0.0]), np.array([[0.0, 0.0], [np.nan, 1.0]]), np.array([[0.0, np.inf], [1.0, 0.0]])]
    for bad in bad_inputs:
        with pytest.raises(ValueError):
            fcn(bad)

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity:
    ∑_{i=1}^8 N_i(ξ,η) = 1 for all (ξ,η) in the reference square [-1,1] × [-1,1].
    This test evaluates ∑ N_i at selected sample points (corners, mid-sides,
    and centroid) and ensures that the sum equals 1 within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    S = np.sum(N[:, :, 0], axis=1)
    assert np.allclose(S, np.ones(pts.shape[0]), rtol=1e-12, atol=1e-12)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies ∑_i ∇N_i(ξ,η) = (0,0) for all (ξ,η) in [-1,1]×[-1,1].
    This test evaluates the gradient sum at selected points (corners, mid-sides, centroid).
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    pts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, dN) = fcn(pts)
    grad_sum = np.sum(dN, axis=1)
    assert np.allclose(grad_sum, np.zeros_like(grad_sum), rtol=1e-12, atol=1e-12)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """For Q8 quadrilaterals, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each of the 8 reference nodes and assembles an 8×8 matrix
    whose (i,j) entry is N_i at node_j. The matrix should equal the identity."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, dN) = fcn(nodes)
    M = N[:, :, 0]
    assert np.allclose(M, np.eye(8), rtol=1e-12, atol=1e-12)

def test_value_completeness_quad8(fcn):
    """Check that quadratic (Q8) quad shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.25, -0.5], [-0.6, 0.4]])
    (N_nodes, _) = fcn(nodes)
    (N_samp, _) = fcn(samples)
    N_nodes_mat = N_nodes[:, :, 0]
    N_samp_mat = N_samp[:, :, 0]
    xi_nodes = nodes[:, 0]
    eta_nodes = nodes[:, 1]
    xi_s = samples[:, 0]
    eta_s = samples[:, 1]
    polys = [lambda x, y: np.ones_like(x), lambda x, y: x, lambda x, y: y, lambda x, y: x * y, lambda x, y: x ** 2, lambda x, y: y ** 2]
    for p in polys:
        v_nodes = p(xi_nodes, eta_nodes)
        interp = N_samp_mat @ v_nodes
        exact = p(xi_s, eta_s)
        assert np.allclose(interp, exact, rtol=1e-12, atol=1e-12)

def test_gradient_completeness_quad8(fcn):
    """Check that Q8 quad shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0], [0.25, -0.5], [-0.6, 0.4]])
    (N_nodes, _) = fcn(nodes)
    (N_samp, dN_samp) = fcn(samples)
    xi_nodes = nodes[:, 0]
    eta_nodes = nodes[:, 1]
    xi_s = samples[:, 0]
    eta_s = samples[:, 1]
    polynomials = [(lambda x, y: np.ones_like(x), lambda x, y: (np.zeros_like(x), np.zeros_like(y))), (lambda x, y: x, lambda x, y: (np.ones_like(x), np.zeros_like(y))), (lambda x, y: y, lambda x, y: (np.zeros_like(x), np.ones_like(y))), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: x ** 2, lambda x, y: (2 * x, np.zeros_like(y))), (lambda x, y: y ** 2, lambda x, y: (np.zeros_like(x), 2 * y))]
    for (p, gradp) in polynomials:
        v_nodes = p(xi_nodes, eta_nodes)
        grad_pred = np.einsum('i,mij->mj', v_nodes, dN_samp)
        (gx_exact, gy_exact) = gradp(xi_s, eta_s)
        grad_exact = np.stack([gx_exact, gy_exact], axis=1)
        assert np.allclose(grad_pred, grad_exact, rtol=1e-12, atol=1e-12)