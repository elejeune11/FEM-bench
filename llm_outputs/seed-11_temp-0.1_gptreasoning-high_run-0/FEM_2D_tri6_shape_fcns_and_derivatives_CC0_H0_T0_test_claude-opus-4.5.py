def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    import pytest
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn((0.5, 0.5))
    with pytest.raises(ValueError):
        fcn(np.array([0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.5, 0.5]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([0.5, -np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [np.nan, 0.25]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.1], [0.2, 0.6]])
    (N, _) = fcn(sample_points)
    N_sum = np.sum(N, axis=1)
    assert np.allclose(N_sum, 1.0, atol=1e-14), f'Partition of unity failed: {N_sum}'

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3], [0.25, 0.25], [0.1, 0.7]])
    (_, dN_dxi) = fcn(sample_points)
    dN_sum = np.sum(dN_dxi, axis=1)
    assert np.allclose(dN_sum, 0.0, atol=1e-14), f'Derivative partition of unity failed: {dN_sum}'

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.squeeze(-1)
    identity = np.eye(6)
    assert np.allclose(N_matrix, identity, atol=1e-14), f'Kronecker delta property failed:\n{N_matrix}'

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.1], [0.3, 0.4], [0.2, 0.6], [1 / 3, 1 / 3]])

    def poly_constant(xi, eta):
        return np.ones_like(xi)

    def poly_linear_xi(xi, eta):
        return xi

    def poly_linear_eta(xi, eta):
        return eta

    def poly_quad_xi2(xi, eta):
        return xi ** 2

    def poly_quad_eta2(xi, eta):
        return eta ** 2

    def poly_quad_xieta(xi, eta):
        return xi * eta
    polynomials = [poly_constant, poly_linear_xi, poly_linear_eta, poly_quad_xi2, poly_quad_eta2, poly_quad_xieta]
    (N_sample, _) = fcn(sample_points)
    for poly in polynomials:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        interpolated = np.sum(N_sample[:, :, 0] * nodal_values, axis=1)
        exact = poly(sample_points[:, 0], sample_points[:, 1])
        max_error = np.max(np.abs(interpolated - exact))
        assert max_error < 1e-14, f'Value completeness failed for polynomial with max error {max_error}'

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.1], [0.3, 0.4], [0.2, 0.6], [1 / 3, 1 / 3]])
    test_cases = [(lambda xi, eta: np.ones_like(xi), lambda xi, eta: np.zeros_like(xi), lambda xi, eta: np.zeros_like(xi)), (lambda xi, eta: xi, lambda xi, eta: np.ones_like(xi), lambda xi, eta: np.zeros_like(xi)), (lambda xi, eta: eta, lambda xi, eta: np.zeros_like(xi), lambda xi, eta: np.ones_like(xi)), (lambda xi, eta: xi ** 2, lambda xi, eta: 2 * xi, lambda xi, eta: np.zeros_like(xi)), (lambda xi, eta: eta ** 2, lambda xi, eta: np.zeros_like(xi), lambda xi, eta: 2 * eta), (lambda xi, eta: xi * eta, lambda xi, eta: eta, lambda xi, eta: xi)]
    (_, dN_dxi_sample) = fcn(sample_points)
    for (poly, grad_xi, grad_eta) in test_cases:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        interp_grad_xi = np.sum(dN_dxi_sample[:, :, 0] * nodal_values, axis=1)
        interp_grad_eta = np.sum(dN_dxi_sample[:, :, 1] * nodal_values, axis=1)
        exact_grad_xi = grad_xi(sample_points[:, 0], sample_points[:, 1])
        exact_grad_eta = grad_eta(sample_points[:, 0], sample_points[:, 1])
        max_error_xi = np.max(np.abs(interp_grad_xi - exact_grad_xi))
        max_error_eta = np.max(np.abs(interp_grad_eta - exact_grad_eta))
        assert max_error_xi < 1e-14, f'Gradient completeness (xi) failed with max error {max_error_xi}'
        assert max_error_eta < 1e-14, f'Gradient completeness (eta) failed with max error {max_error_eta}'