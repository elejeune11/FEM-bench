def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.5]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, np.nan]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, np.inf]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3], [0.1, np.nan]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.4, 0.1], [0.1, 0.4]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.4, 0.1], [0.1, 0.4]])
    (_, dN_dxi) = fcn(sample_points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(6)
    assert np.allclose(N.squeeze(), identity_matrix, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N_sample, _) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    poly_coeffs = [lambda x, y: 1.0 + 2 * x + 3 * y, lambda x, y: x ** 2 + 2 * x * y + y ** 2]
    for poly in poly_coeffs:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes]).reshape(6, 1)
        exact_values = np.array([poly(pt[0], pt[1]) for pt in sample_points]).reshape(-1, 1)
        interpolated_values = N_sample @ nodal_values
        assert np.max(np.abs(interpolated_values - exact_values)) < 1e-12

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (_, dN_dxi_sample) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    poly_grads = [lambda x, y: (2.0, 3.0), lambda x, y: (2 * x + 2 * y, 2 * x + 2 * y)]
    for poly_grad in poly_grads:
        nodal_values = np.array([poly_grad(node[0], node[1])[0] + poly_grad(node[0], node[1])[1] for node in nodes]).reshape(6, 1)
        exact_grads = np.array([poly_grad(pt[0], pt[1]) for pt in sample_points])
        interpolated_grads = np.einsum('nij,j->ni', dN_dxi_sample, nodal_values.flatten())
        assert np.max(np.abs(interpolated_grads - exact_grads)) < 1e-12