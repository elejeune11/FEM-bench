def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """
    D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,)
    or (n,2) with finite values. Invalid inputs should raise ValueError.
    This test tries a handful of invalid inputs (wrong type, wrong shape, NaNs) and asserts
    that ValueError is raised.
    """
    with pytest.raises(ValueError):
        fcn([0.5, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.5, 0.3]]]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.3, 0.1]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, 0.3], [np.inf, 0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.5, -np.inf]))

def test_partition_of_unity_tri6(fcn):
    """
    Shape functions on a triangle must satisfy the partition of unity:
    ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle.
    This test evaluates ∑ N_i at well considered sample points and ensures 
    that the sum equals 1 within tight tolerance.
    """
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.6, 0.3]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-14, rtol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """
    Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero
    gradient after interpolation. This test evaluates ∑ ∇N_i at canonical sample points.
    For every sample point, the vector sum equals (0,0) within tight tolerance.
    """
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1.0 / 3.0, 1.0 / 3.0], [0.25, 0.25], [0.1, 0.7], [0.6, 0.3]])
    (_, dN_dxi) = fcn(sample_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    np.testing.assert_allclose(grad_sums, np.zeros((len(sample_points), 2)), atol=1e-14, rtol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """
    For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others.
    This test evaluates N at each reference node location and assembles a 6×6 matrix whose
    (i,j) entry is N_i at node_j. This should equal the identity within tight tolerance.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    N_matrix = N[:, :, 0].T
    np.testing.assert_allclose(N_matrix, np.eye(6), atol=1e-14, rtol=1e-14)

def test_value_completeness_tri6(fcn):
    """
    Check that quadratic (P2) triangle shape functions exactly reproduce
    degree-1 and degree-2 polynomials. Nodal values are set from the exact
    polynomial, the field is interpolated at sample points, and the maximum
    error is verified to be nearly zero.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.5], [0.6, 0.2], [0.1, 0.1]])
    nodal_values_deg1 = 1.0 + 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
    exact_deg1 = 1.0 + 2.0 * sample_points[:, 0] + 3.0 * sample_points[:, 1]
    (N, _) = fcn(sample_points)
    interpolated_deg1 = np.sum(N[:, :, 0] * nodal_values_deg1[np.newaxis, :], axis=1)
    np.testing.assert_allclose(interpolated_deg1, exact_deg1, atol=1e-13, rtol=1e-13)
    nodal_values_deg2 = 1.0 + nodes[:, 0] + nodes[:, 1] + nodes[:, 0] ** 2 + nodes[:, 0] * nodes[:, 1] + nodes[:, 1] ** 2
    exact_deg2 = 1.0 + sample_points[:, 0] + sample_points[:, 1] + sample_points[:, 0] ** 2 + sample_points[:, 0] * sample_points[:, 1] + sample_points[:, 1] ** 2
    (N, _) = fcn(sample_points)
    interpolated_deg2 = np.sum(N[:, :, 0] * nodal_values_deg2[np.newaxis, :], axis=1)
    np.testing.assert_allclose(interpolated_deg2, exact_deg2, atol=1e-13, rtol=1e-13)

def test_gradient_completeness_tri6(fcn):
    """
    Check that P2 triangle shape functions reproduce the exact gradient for
    degree-1 and degree-2 polynomials. Gradients are reconstructed from nodal
    values and compared with the analytic gradient at sample points, with
    maximum error verified to be nearly zero.
    """
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.5], [0.6, 0.2], [0.1, 0.1]])
    nodal_values_deg1 = 1.0 + 2.0 * nodes[:, 0] + 3.0 * nodes[:, 1]
    exact_grad_deg1 = np.tile([2.0, 3.0], (len(sample_points), 1))
    (_, dN_dxi) = fcn(sample_points)
    interpolated_grad_deg1 = np.sum(dN_dxi * nodal_values_deg1[np.newaxis, :, np.newaxis], axis=1)
    np.testing.assert_allclose(interpolated_grad_deg1, exact_grad_deg1, atol=1e-13, rtol=1e-13)
    nodal_values_deg2 = 1.0 + nodes[:, 0] + nodes[:, 1] + nodes[:, 0] ** 2 + nodes[:, 0] * nodes[:, 1] + nodes[:, 1] ** 2
    exact_grad_deg2 = np.column_stack([1.0 + 2.0 * sample_points[:, 0] + sample_points[:, 1], 1.0 + sample_points[:, 0] + 2.0 * sample_points[:, 1]])
    (_, dN_dxi) = fcn(sample_points)
    interpolated_grad_deg2 = np.sum(dN_dxi * nodal_values_deg2[np.newaxis, :, np.newaxis], axis=1)
    np.testing.assert_allclose(interpolated_grad_deg2, exact_grad_deg2, atol=1e-13, rtol=1e-13)