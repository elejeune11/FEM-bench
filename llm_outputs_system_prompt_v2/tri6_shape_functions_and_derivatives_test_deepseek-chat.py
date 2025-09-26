def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, np.nan]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, np.inf]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.4, 0.4], [0.8, 0.1]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.4, 0.4], [0.8, 0.1]])
    (_, dN_dxi) = fcn(sample_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(6)
    assert np.allclose(N.squeeze(), identity_matrix, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.4, 0.4], [0.8, 0.1]])
    (N_sample, _) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    poly1 = lambda xi: 2.0 + 3.0 * xi[0] + 4.0 * xi[1]
    poly2 = lambda xi: 1.0 + 2.0 * xi[0] + 3.0 * xi[1] + 4.0 * xi[0] ** 2 + 5.0 * xi[0] * xi[1] + 6.0 * xi[1] ** 2
    nodal_values_p1 = np.array([poly1(node) for node in nodes]).reshape(6, 1)
    nodal_values_p2 = np.array([poly2(node) for node in nodes]).reshape(6, 1)
    interpolated_p1 = N_sample @ nodal_values_p1
    exact_p1 = np.array([poly1(point) for point in sample_points]).reshape(-1, 1)
    error_p1 = np.max(np.abs(interpolated_p1 - exact_p1))
    interpolated_p2 = N_sample @ nodal_values_p2
    exact_p2 = np.array([poly2(point) for point in sample_points]).reshape(-1, 1)
    error_p2 = np.max(np.abs(interpolated_p2 - exact_p2))
    assert error_p1 < 1e-14
    assert error_p2 < 1e-14

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.4, 0.4], [0.8, 0.1]])
    (_, dN_dxi_sample) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    poly1 = lambda xi: 2.0 + 3.0 * xi[0] + 4.0 * xi[1]
    grad_poly1 = lambda xi: np.array([3.0, 4.0])
    poly2 = lambda xi: 1.0 + 2.0 * xi[0] + 3.0 * xi[1] + 4.0 * xi[0] ** 2 + 5.0 * xi[0] * xi[1] + 6.0 * xi[1] ** 2
    grad_poly2 = lambda xi: np.array([2.0 + 8.0 * xi[0] + 5.0 * xi[1], 3.0 + 5.0 * xi[0] + 12.0 * xi[1]])
    nodal_values_p1 = np.array([poly1(node) for node in nodes]).reshape(6, 1)
    nodal_values_p2 = np.array([poly2(node) for node in nodes]).reshape(6, 1)
    grad_interpolated_p1 = np.einsum('nij,j->ni', dN_dxi_sample, nodal_values_p1.squeeze())
    exact_grad_p1 = np.array([grad_poly1(point) for point in sample_points])
    error_grad_p1 = np.max(np.abs(grad_interpolated_p1 - exact_grad_p1))
    grad_interpolated_p2 = np.einsum('nij,j->ni', dN_dxi_sample, nodal_values_p2.squeeze())
    exact_grad_p2 = np.array([grad_poly2(point) for point in sample_points])
    error_grad_p2 = np.max(np.abs(grad_interpolated_p2 - exact_grad_p2))
    assert error_grad_p1 < 1e-14
    assert error_grad_p2 < 1e-14