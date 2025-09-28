def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError. This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn((0.2, 0.3))
    with pytest.raises(ValueError):
        fcn(np.array([0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3], [np.nan, 0.5]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle. This test evaluates ∑ N_i at well considered sample points and ensures that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.25, 0.25], [0.3, 0.3], [0.1, 0.2]])
    (N, _) = fcn(sample_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points. For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.25, 0.25], [0.3, 0.3], [0.1, 0.2]])
    (_, dN_dxi) = fcn(sample_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others. This test evaluates N at each reference node location and assembles a 6×6 matrix whose (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(node_coords)
    N_matrix = N.reshape(6, 6)
    identity_matrix = np.eye(6)
    assert np.allclose(N_matrix, identity_matrix, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials. Nodal values are set from the exact polynomial, the field is interpolated at sample points, and the maximum error is verified to be nearly zero."""
    test_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3], [0.25, 0.25]])
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N_test, _) = fcn(test_points)
    (N_nodes, _) = fcn(node_coords)
    linear_polys = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta]
    quadratic_polys = [lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi * eta]
    all_polys = linear_polys + quadratic_polys
    for poly in all_polys:
        nodal_values = np.array([poly(xi, eta) for (xi, eta) in node_coords])
        interpolated_values = np.dot(N_test.reshape(len(test_points), 6), nodal_values)
        exact_values = np.array([poly(xi, eta) for (xi, eta) in test_points])
        max_error = np.max(np.abs(interpolated_values - exact_values))
        assert max_error < 1e-12

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal values and compared with the analytic gradient at sample points, with maximum error verified to be nearly zero."""
    test_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3], [0.25, 0.25]])
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (_, dN_test) = fcn(test_points)
    (N_nodes, _) = fcn(node_coords)
    linear_cases = [(lambda xi, eta: 1.0, lambda xi, eta: (0.0, 0.0)), (lambda xi, eta: xi, lambda xi, eta: (1.0, 0.0)), (lambda xi, eta: eta, lambda xi, eta: (0.0, 1.0))]
    quadratic_cases = [(lambda xi, eta: xi ** 2, lambda xi, eta: (2 * xi, 0.0)), (lambda xi, eta: eta ** 2, lambda xi, eta: (0.0, 2 * eta)), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi))]
    all_cases = linear_cases + quadratic_cases
    for (poly, grad_poly) in all_cases:
        nodal_values = np.array([poly(xi, eta) for (xi, eta) in node_coords])
        reconstructed_gradients = np.zeros((len(test_points), 2))
        for i in range(6):
            reconstructed_gradients[:, 0] += dN_test[:, i, 0] * nodal_values[i]
            reconstructed_gradients[:, 1] += dN_test[:, i, 1] * nodal_values[i]
        exact_gradients = np.array([grad_poly(xi, eta) for (xi, eta) in test_points])
        max_error = np.max(np.abs(reconstructed_gradients - exact_gradients))
        assert max_error < 1e-12