def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    invalid_inputs = [[0.5, 0.5], np.array([0.5]), np.array([[0.5, 0.5], [0.5]]), np.array([0.5, np.nan]), np.array([0.5, np.inf])]
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(invalid_input)

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3]])
    (N, _) = fcn(sample_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [1 / 3, 1 / 3]])
    (_, dN_dxi) = fcn(sample_points)
    sum_dN_dxi = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN_dxi, np.zeros((len(sample_points), 2)), atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    node_locations = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(node_locations)
    N_matrix = N.squeeze(axis=2)
    identity_matrix = np.eye(6)
    assert np.allclose(N_matrix, identity_matrix, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def polynomial(xi, eta):
        return 1 + xi + eta + xi * eta + xi ** 2 + eta ** 2
    node_locations = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    nodal_values = polynomial(node_locations[:, 0], node_locations[:, 1])
    sample_points = np.array([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5], [1 / 3, 1 / 3]])
    (N, _) = fcn(sample_points)
    interpolated_values = np.dot(N.squeeze(axis=2), nodal_values)
    exact_values = polynomial(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(interpolated_values, exact_values, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def polynomial_gradient(xi, eta):
        df_dxi = 1 + eta + 2 * xi
        df_deta = 1 + xi + 2 * eta
        return np.array([df_dxi, df_deta])
    node_locations = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    sample_points = np.array([[0.25, 0.25], [0.5, 0.25], [0.25, 0.5], [1 / 3, 1 / 3]])
    (_, dN_dxi) = fcn(sample_points)
    nodal_values = np.array([polynomial_gradient(xi, eta) for (xi, eta) in node_locations])
    interpolated_gradients = np.einsum('ijk,ik->ij', dN_dxi, nodal_values)
    exact_gradients = np.array([polynomial_gradient(xi, eta) for (xi, eta) in sample_points])
    assert np.allclose(interpolated_gradients, exact_gradients, atol=1e-12)