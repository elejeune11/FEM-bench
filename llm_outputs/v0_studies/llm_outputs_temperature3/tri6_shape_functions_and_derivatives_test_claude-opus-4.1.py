def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn(np.array([0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.5, 0.5]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.5], [np.nan, 0.2]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    test_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.1, 0.7], [0.6, 0.2]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1).flatten()
    assert np.allclose(sums, 1.0, rtol=1e-12, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3], [0.2, 0.3], [0.1, 0.7], [0.6, 0.2]])
    (_, dN_dxi) = fcn(test_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, rtol=1e-12, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    kronecker_matrix = N[:, :, 0].T
    expected = np.eye(6)
    assert np.allclose(kronecker_matrix, expected, rtol=1e-12, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.2, 0.3], [0.1, 0.7], [0.6, 0.2], [1 / 3, 1 / 3]])
    test_polys = [lambda xi, eta: np.ones_like(xi), lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: xi ** 2, lambda xi, eta: xi * eta, lambda xi, eta: eta ** 2]
    (N, _) = fcn(test_points)
    for poly in test_polys:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        interpolated = np.sum(N * nodal_values[np.newaxis, :, np.newaxis], axis=1).flatten()
        exact = poly(test_points[:, 0], test_points[:, 1])
        assert np.allclose(interpolated, exact, rtol=1e-12, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    test_points = np.array([[0.2, 0.3], [0.1, 0.7], [0.6, 0.2], [1 / 3, 1 / 3]])
    test_cases = [(lambda xi, eta: np.ones_like(xi), lambda xi, eta: (np.zeros_like(xi), np.zeros_like(eta))), (lambda xi, eta: xi, lambda xi, eta: (np.ones_like(xi), np.zeros_like(eta))), (lambda xi, eta: eta, lambda xi, eta: (np.zeros_like(xi), np.ones_like(eta))), (lambda xi, eta: xi ** 2, lambda xi, eta: (2 * xi, np.zeros_like(eta))), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi)), (lambda xi, eta: eta ** 2, lambda xi, eta: (np.zeros_like(xi), 2 * eta))]
    (_, dN_dxi) = fcn(test_points)
    for (poly, grad_exact) in test_cases:
        nodal_values = poly(nodes[:, 0], nodes[:, 1])
        grad_interp = np.sum(dN_dxi * nodal_values[np.newaxis, :, np.newaxis], axis=1)
        (grad_xi_exact, grad_eta_exact) = grad_exact(test_points[:, 0], test_points[:, 1])
        grad_exact_array = np.column_stack([grad_xi_exact, grad_eta_exact])
        assert np.allclose(grad_interp, grad_exact_array, rtol=1e-12, atol=1e-12)