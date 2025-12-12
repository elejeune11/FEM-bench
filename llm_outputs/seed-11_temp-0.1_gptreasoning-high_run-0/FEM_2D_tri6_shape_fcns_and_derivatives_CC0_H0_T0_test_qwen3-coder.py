def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    with pytest.raises(ValueError):
        fcn([[0.1, 0.2]])
    with pytest.raises(ValueError):
        fcn(np.array([0.1]))
    with pytest.raises(ValueError):
        fcn(np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2], [np.nan, 0.4]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.1, 0.2], [0.3, np.inf]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    test_points = np.array([[0.1, 0.2], [0.3, 0.3], [0.5, 0.1], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.33, 0.33]])
    (N, _) = fcn(test_points)
    sums = np.sum(N, axis=1).squeeze()
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    test_points = np.array([[0.1, 0.2], [0.3, 0.3], [0.5, 0.1], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.33, 0.33]])
    (_, dN_dxi) = fcn(test_points)
    sums = np.sum(dN_dxi, axis=1)
    expected = np.zeros_like(sums)
    np.testing.assert_allclose(sums, expected, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(node_coords)
    N_matrix = N.squeeze()
    expected = np.eye(6)
    np.testing.assert_allclose(N_matrix, expected, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def poly(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    nodal_values = poly(node_coords[:, 0], node_coords[:, 1])
    nodal_values = nodal_values[:, np.newaxis]
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.33, 0.33]])
    (N, _) = fcn(sample_points)
    interpolated = np.sum(N * nodal_values.T, axis=1)
    interpolated = interpolated.squeeze()
    exact = poly(sample_points[:, 0], sample_points[:, 1])
    np.testing.assert_allclose(interpolated, exact, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def poly(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2

    def grad_poly(x, y):
        df_dx = 2 + 8 * x + 5 * y
        df_dy = 3 + 5 * x + 12 * y
        return np.stack([df_dx, df_dy], axis=-1)
    node_coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    nodal_values = poly(node_coords[:, 0], node_coords[:, 1])
    nodal_values = nodal_values[:, np.newaxis]
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.33, 0.33]])
    (_, dN_dxi) = fcn(sample_points)
    reconstructed_grad = np.sum(dN_dxi * nodal_values.T, axis=1)
    exact_grad = grad_poly(sample_points[:, 0], sample_points[:, 1])
    np.testing.assert_allclose(reconstructed_grad, exact_grad, atol=1e-12)