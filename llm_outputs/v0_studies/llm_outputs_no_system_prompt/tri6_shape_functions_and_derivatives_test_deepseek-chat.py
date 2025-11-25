def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn((0.2, 0.3))
    with pytest.raises(ValueError):
        fcn(np.array([0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError):
        fcn(np.array([[[0.2, 0.3]]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, np.nan]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, -np.inf]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.3, 0.6], [0.15, 0.25]])
    (N, _) = fcn(sample_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.3, 0.6], [0.15, 0.25]])
    (_, dN_dxi) = fcn(sample_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N, _) = fcn(nodes)
    N_matrix = N.reshape(6, 6)
    assert np.allclose(N_matrix, np.eye(6), atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.3, 0.6], [0.15, 0.25]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N_nodes, _) = fcn(nodes)
    (N_samples, _) = fcn(sample_points)
    linear_polys = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta]
    quadratic_polys = [lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi * eta]
    max_error = 0.0
    for poly in linear_polys + quadratic_polys:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes]).reshape(6, 1)
        interpolated = np.einsum('nij,jk->nik', N_samples, nodal_values).reshape(-1)
        exact = np.array([poly(point[0], point[1]) for point in sample_points])
        max_error = max(max_error, np.max(np.abs(interpolated - exact)))
    assert max_error < 1e-13

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.5, 0.2], [0.1, 0.1], [0.3, 0.6], [0.15, 0.25]])
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.5], [0.0, 0.5], [0.5, 0.0]])
    (N_nodes, dN_nodes) = fcn(nodes)
    (_, dN_samples) = fcn(sample_points)
    linear_polys = [(lambda xi, eta: 1.0, lambda xi, eta: (0.0, 0.0)), (lambda xi, eta: xi, lambda xi, eta: (1.0, 0.0)), (lambda xi, eta: eta, lambda xi, eta: (0.0, 1.0))]
    quadratic_polys = [(lambda xi, eta: xi ** 2, lambda xi, eta: (2 * xi, 0.0)), (lambda xi, eta: eta ** 2, lambda xi, eta: (0.0, 2 * eta)), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi))]
    max_error = 0.0
    for (poly, grad_poly) in linear_polys + quadratic_polys:
        nodal_values = np.array([poly(node[0], node[1]) for node in nodes]).reshape(6, 1)
        reconstructed_grad = np.einsum('nij,jk->nik', dN_samples, nodal_values).reshape(-1, 2)
        exact_grad = np.array([grad_poly(point[0], point[1]) for point in sample_points])
        max_error = max(max_error, np.max(np.abs(reconstructed_grad - exact_grad)))
    assert max_error < 1e-12