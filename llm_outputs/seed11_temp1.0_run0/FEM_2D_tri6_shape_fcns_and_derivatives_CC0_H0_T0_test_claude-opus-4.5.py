def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.25, 0.25])
    with pytest.raises(ValueError):
        fcn((0.25, 0.25))
    with pytest.raises(ValueError):
        fcn(np.array([0.25, 0.25, 0.25]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.25, 0.25]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.25, 0.25, 0.25], [0.1, 0.1, 0.1]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.25]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.25]))
    with pytest.raises(ValueError):
        fcn(np.array([0.25, -np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.25, 0.25], [np.nan, 0.1]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.1], [0.6, 0.2]])
    (N, dN_dxi) = fcn(sample_points)
    N_sum = np.sum(N, axis=1)
    assert np.allclose(N_sum, 1.0, atol=1e-14), f'Partition of unity failed: max error = {np.max(np.abs(N_sum - 1.0))}'

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7]])
    (N, dN_dxi) = fcn(sample_points)
    dN_sum = np.sum(dN_dxi, axis=1)
    assert np.allclose(dN_sum, 0.0, atol=1e-14), f'Derivative partition of unity failed: max error = {np.max(np.abs(dN_sum))}'

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, dN_dxi) = fcn(nodes)
    N_matrix = N[:, :, 0]
    expected = np.eye(6)
    assert np.allclose(N_matrix, expected, atol=1e-14), f'Kronecker delta property failed: max error = {np.max(np.abs(N_matrix - expected))}'

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.1], [0.3, 0.5], [0.2, 0.6], [1 / 3, 1 / 3]])

    def poly_const(x):
        return np.ones(len(x))

    def poly_xi(x):
        return x[:, 0]

    def poly_eta(x):
        return x[:, 1]

    def poly_xi2(x):
        return x[:, 0] ** 2

    def poly_eta2(x):
        return x[:, 1] ** 2

    def poly_xieta(x):
        return x[:, 0] * x[:, 1]
    polynomials = [poly_const, poly_xi, poly_eta, poly_xi2, poly_eta2, poly_xieta]
    (N, _) = fcn(sample_points)
    for poly in polynomials:
        nodal_values = poly(nodes)
        interpolated = np.sum(N[:, :, 0] * nodal_values, axis=1)
        exact = poly(sample_points)
        assert np.allclose(interpolated, exact, atol=1e-13), f'Value completeness failed for polynomial'

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.1], [0.3, 0.5], [0.2, 0.6], [1 / 3, 1 / 3]])
    test_cases = [(lambda x: x[:, 0], lambda x: np.ones(len(x)), lambda x: np.zeros(len(x))), (lambda x: x[:, 1], lambda x: np.zeros(len(x)), lambda x: np.ones(len(x))), (lambda x: x[:, 0] ** 2, lambda x: 2 * x[:, 0], lambda x: np.zeros(len(x))), (lambda x: x[:, 1] ** 2, lambda x: np.zeros(len(x)), lambda x: 2 * x[:, 1]), (lambda x: x[:, 0] * x[:, 1], lambda x: x[:, 1], lambda x: x[:, 0])]
    (_, dN_dxi) = fcn(sample_points)
    for (poly, grad_xi_exact, grad_eta_exact) in test_cases:
        nodal_values = poly(nodes)
        grad_xi_interp = np.sum(dN_dxi[:, :, 0] * nodal_values, axis=1)
        grad_eta_interp = np.sum(dN_dxi[:, :, 1] * nodal_values, axis=1)
        exact_grad_xi = grad_xi_exact(sample_points)
        exact_grad_eta = grad_eta_exact(sample_points)
        assert np.allclose(grad_xi_interp, exact_grad_xi, atol=1e-13), 'Gradient completeness failed for d/dxi'
        assert np.allclose(grad_eta_interp, exact_grad_eta, atol=1e-13), 'Gradient completeness failed for d/deta'