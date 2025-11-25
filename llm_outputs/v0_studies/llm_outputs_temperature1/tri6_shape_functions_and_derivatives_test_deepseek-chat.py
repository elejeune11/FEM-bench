def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.3]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.2, 0.3], [0.1, 0.1], [0.4, 0.2]])
    (N, _) = fcn(sample_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [0.2, 0.3], [0.1, 0.1], [0.4, 0.2]])
    (_, dN_dxi) = fcn(sample_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(6, 6)
    assert np.allclose(N_matrix, np.eye(6), atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.1, 0.1], [0.4, 0.2], [0.3, 0.1], [0.1, 0.4]])

    def linear_func(xi):
        return 2.0 + 3.0 * xi[:, 0] + 1.5 * xi[:, 1]

    def quadratic_func(xi):
        (ξ, η) = (xi[:, 0], xi[:, 1])
        return 2.0 + 3.0 * ξ + 1.5 * η + 0.5 * ξ ** 2 + 0.8 * η ** 2 + 1.2 * ξ * η
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N_sample, _) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    nodal_values_linear = linear_func(nodes).reshape(-1, 1)
    interpolated_linear = np.sum(N_sample * nodal_values_linear.T, axis=2)
    exact_linear = linear_func(sample_points)
    assert np.max(np.abs(interpolated_linear.flatten() - exact_linear)) < 1e-12
    nodal_values_quad = quadratic_func(nodes).reshape(-1, 1)
    interpolated_quad = np.sum(N_sample * nodal_values_quad.T, axis=2)
    exact_quad = quadratic_func(sample_points)
    assert np.max(np.abs(interpolated_quad.flatten() - exact_quad)) < 1e-12

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.1, 0.1], [0.4, 0.2], [0.3, 0.1], [0.1, 0.4]])

    def linear_grad(xi):
        return np.full((xi.shape[0], 2), [3.0, 1.5])

    def quadratic_grad(xi):
        (ξ, η) = (xi[:, 0], xi[:, 1])
        grad_ξ = 3.0 + 2 * 0.5 * ξ + 1.2 * η
        grad_η = 1.5 + 2 * 0.8 * η + 1.2 * ξ
        return np.column_stack([grad_ξ, grad_η])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N_sample, dN_dxi_sample) = fcn(sample_points)
    (_, dN_dxi_nodes) = fcn(nodes)
    nodal_values_linear = (2.0 + 3.0 * nodes[:, 0] + 1.5 * nodes[:, 1]).reshape(-1, 1)
    interpolated_grad_linear = np.sum(dN_dxi_sample * nodal_values_linear.T[:, :, np.newaxis], axis=1)
    exact_grad_linear = linear_grad(sample_points)
    assert np.max(np.abs(interpolated_grad_linear - exact_grad_linear)) < 1e-12
    nodal_values_quad = (2.0 + 3.0 * nodes[:, 0] + 1.5 * nodes[:, 1] + 0.5 * nodes[:, 0] ** 2 + 0.8 * nodes[:, 1] ** 2 + 1.2 * nodes[:, 0] * nodes[:, 1]).reshape(-1, 1)
    interpolated_grad_quad = np.sum(dN_dxi_sample * nodal_values_quad.T[:, :, np.newaxis], axis=1)
    exact_grad_quad = quadratic_grad(sample_points)
    assert np.max(np.abs(interpolated_grad_quad - exact_grad_quad)) < 1e-12