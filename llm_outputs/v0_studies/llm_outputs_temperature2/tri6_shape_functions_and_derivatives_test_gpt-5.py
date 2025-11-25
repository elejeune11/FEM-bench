def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError. This test tries
    a handful of invalid inputs (wrong type, wrong shape, NaNs, Infs) and asserts that
    ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn((0.2, 0.3))
    with pytest.raises(ValueError):
        fcn(0.5)
    with pytest.raises(ValueError):
        fcn(np.array(0.5))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2], [0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.1]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, np.nan], [0.1, 0.1]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3], [np.inf, 0.0]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    sum_{i=1}^6 N_i(xi,eta) = 1 for all (xi,eta) in the reference triangle.
    This test evaluates sum N_i at representative sample points and ensures
    that the sum equals 1 within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.1, 0.2], [0.6, 0.2]])
    (N, _) = fcn(pts)
    sums = N.sum(axis=1).reshape(-1)
    assert np.allclose(sums, np.ones_like(sums), atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies sum_i grad N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates the vector sum at representative
    sample points and checks it equals (0,0) within tight tolerance."""
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.5, 0.0], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.2, 0.3], [0.1, 0.2], [0.6, 0.2]])
    (_, dN) = fcn(pts)
    grad_sums = dN.sum(axis=1)
    assert np.allclose(grad_sums, np.zeros_like(grad_sums), atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6x6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    M = N[..., 0]
    I = np.eye(6)
    assert M.shape == I.shape
    assert np.allclose(M, I, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and
    degree-2 polynomials. Nodal values are set from the exact polynomial, the field is
    interpolated at sample points, and the maximum error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0], [0.6, 0.2]])
    polynomials = [lambda x, y: 1.0 + 2.0 * x - 3.0 * y, lambda x, y: x * x, lambda x, y: y * y, lambda x, y: x * y, lambda x, y: x * x + y * y + x * y + x + y + 1.0]
    (N_nodes, _) = fcn(nodes)
    N_nodes = N_nodes[..., 0]
    (N_samp, _) = fcn(samples)
    N_samp = N_samp[..., 0]
    for p in polynomials:
        vals_nodes = p(nodes[:, 0], nodes[:, 1])
        approx = N_samp @ vals_nodes
        exact = p(samples[:, 0], samples[:, 1])
        assert np.allclose(approx, exact, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1
    and degree-2 polynomials. Gradients are reconstructed from nodal values and compared
    with the analytic gradient at sample points, with maximum error nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    samples = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0], [0.2, 0.0], [0.0, 0.2], [0.2, 0.3], [1.0 / 3.0, 1.0 / 3.0], [0.6, 0.2]])
    funcs = [(lambda x, y: 1.0 + 2.0 * x - 3.0 * y, lambda x, y: (2.0, -3.0)), (lambda x, y: x * x, lambda x, y: (2.0 * x, 0.0)), (lambda x, y: y * y, lambda x, y: (0.0, 2.0 * y)), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: x * x + y * y + x * y + x + y + 1.0, lambda x, y: (2.0 * x + y + 1.0, 2.0 * y + x + 1.0))]
    (N_nodes, _) = fcn(nodes)
    N_nodes = N_nodes[..., 0]
    (_, dN_samp) = fcn(samples)
    for (val_fn, grad_fn) in funcs:
        vals_nodes = val_fn(nodes[:, 0], nodes[:, 1])
        grad_xi = dN_samp[:, :, 0] @ vals_nodes
        grad_eta = dN_samp[:, :, 1] @ vals_nodes
        (gx_exact, gy_exact) = grad_fn(samples[:, 0], samples[:, 1])
        gx_exact = np.asarray(gx_exact)
        gy_exact = np.asarray(gy_exact)
        assert np.allclose(grad_xi, gx_exact, atol=1e-12)
        assert np.allclose(grad_eta, gy_exact, atol=1e-12)