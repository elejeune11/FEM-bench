def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised."""
    invalid_inputs = [[0.5, 0.5], np.array([0.5]), np.array([[0.5, 0.5, 0.5]]), np.array([[0.5, np.nan]]), np.array([[0.5, np.inf]])]
    for invalid_input in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(invalid_input)

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (N, _) = fcn(sample_points)
    for i in range(N.shape[0]):
        assert np.isclose(np.sum(N[i, :, 0]), 1.0, atol=1e-09)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    (_, dN_dxi) = fcn(sample_points)
    for i in range(dN_dxi.shape[0]):
        assert np.allclose(np.sum(dN_dxi[i, :, :], axis=0), np.array([0.0, 0.0]), atol=1e-09)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance."""
    node_points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(node_points)
    identity_matrix = np.eye(6)
    for i in range(6):
        for j in range(6):
            assert np.isclose(N[j, i, 0], identity_matrix[i, j], atol=1e-09)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero."""

    def poly(xi, eta):
        return 1 + 2 * xi + 3 * eta + 4 * xi ** 2 + 5 * xi * eta + 6 * eta ** 2
    node_points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    nodal_values = np.array([poly(xi, eta) for (xi, eta) in node_points])
    sample_points = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    (N, _) = fcn(sample_points)
    for i in range(sample_points.shape[0]):
        interpolated_value = np.sum(N[i, :, 0] * nodal_values)
        exact_value = poly(sample_points[i, 0], sample_points[i, 1])
        assert np.isclose(interpolated_value, exact_value, atol=1e-09)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero."""

    def grad_poly(xi, eta):
        return np.array([2 + 8 * xi + 5 * eta, 3 + 5 * xi + 12 * eta])
    node_points = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    nodal_values = np.array([grad_poly(xi, eta) for (xi, eta) in node_points])
    sample_points = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75]])
    (_, dN_dxi) = fcn(sample_points)
    for i in range(sample_points.shape[0]):
        interpolated_gradient = np.sum(dN_dxi[i, :, :] * nodal_values[:, np.newaxis], axis=0)
        exact_gradient = grad_poly(sample_points[i, 0], sample_points[i, 1])
        assert np.allclose(interpolated_gradient, exact_gradient, atol=1e-09)