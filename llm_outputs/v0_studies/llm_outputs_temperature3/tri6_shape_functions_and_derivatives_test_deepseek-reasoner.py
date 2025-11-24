def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.5, 0.5])
    with pytest.raises(ValueError):
        fcn('invalid')
    with pytest.raises(ValueError):
        fcn(np.array([1.0]))
    with pytest.raises(ValueError):
        fcn(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, 2.0, 3.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.5]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.5, np.nan], [0.2, 0.3]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.25, 0.25], [0.25, 0.5], [0.5, 0.25], [1 / 3, 1 / 3], [0.2, 0.3], [0.1, 0.1]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-12)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.25, 0.25], [1 / 3, 1 / 3], [0.2, 0.3]])
    (_, dN_dxi) = fcn(sample_points)
    derivative_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(derivative_sums, 0.0, atol=1e-12)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(6)
    assert np.allclose(N.reshape(6, 6), identity_matrix, atol=1e-12)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.25, 0.25], [0.5, 0.0], [0.0, 0.5], [0.33, 0.33], [0.2, 0.3], [0.4, 0.1]])
    linear_polys = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: 1.0 + 2 * xi - 3 * eta]
    quadratic_polys = [lambda xi, eta: xi * eta, lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: 1.0 + xi - 2 * eta + 3 * xi * eta - xi ** 2 + 2 * eta ** 2]
    (N, _) = fcn(sample_points)
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    for poly in linear_polys + quadratic_polys:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        interpolated = np.dot(N.reshape(len(sample_points), 6), nodal_values)
        exact = np.array([poly(p[0], p[1]) for p in sample_points])
        assert np.allclose(interpolated, exact, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.25, 0.25], [0.5, 0.0], [0.0, 0.5], [0.33, 0.33], [0.2, 0.3], [0.4, 0.1]])
    test_cases = [(lambda xi, eta: 1.0, lambda xi, eta: [0.0, 0.0]), (lambda xi, eta: xi, lambda xi, eta: [1.0, 0.0]), (lambda xi, eta: eta, lambda xi, eta: [0.0, 1.0]), (lambda xi, eta: 1.0 + 2 * xi - 3 * eta, lambda xi, eta: [2.0, -3.0]), (lambda xi, eta: xi * eta, lambda xi, eta: [eta, xi]), (lambda xi, eta: xi ** 2, lambda xi, eta: [2 * xi, 0.0]), (lambda xi, eta: eta ** 2, lambda xi, eta: [0.0, 2 * eta])]
    (N, dN_dxi) = fcn(sample_points)
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    for (poly, grad_poly) in test_cases:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes])
        reconstructed_grad = np.zeros((len(sample_points), 2))
        for i in range(len(sample_points)):
            for j in range(2):
                reconstructed_grad[i, j] = np.dot(dN_dxi[i, :, j], nodal_values)
        exact_grad = np.array([grad_poly(p[0], p[1]) for p in sample_points])
        assert np.allclose(reconstructed_grad, exact_grad, atol=1e-12)