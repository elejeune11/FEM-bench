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

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.1, 0.8], [0.7, 0.1]])
    (N, _) = fcn(points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.1, 0.8], [0.7, 0.1]])
    (_, dN_dxi) = fcn(points)
    grad_sums = np.sum(dN_dxi, axis=1)
    assert np.allclose(grad_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(6)
    assert np.allclose(N.reshape(6, 6), identity_matrix, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    points = np.array([[0.2, 0.3], [0.1, 0.8], [0.7, 0.1], [0.4, 0.4], [0.25, 0.25]])
    (N, _) = fcn(points)
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    test_polys = [lambda x, y: 1.0, lambda x, y: x, lambda x, y: y, lambda x, y: x * y, lambda x, y: x * x, lambda x, y: y * y]
    for poly in test_polys:
        nodal_vals = np.array([poly(n[0], n[1]) for n in nodes]).reshape(6, 1)
        exact_vals = np.array([poly(p[0], p[1]) for p in points]).reshape(-1, 1)
        interpolated_vals = N @ nodal_vals
        assert np.allclose(interpolated_vals, exact_vals, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    points = np.array([[0.2, 0.3], [0.1, 0.8], [0.7, 0.1], [0.4, 0.4], [0.25, 0.25]])
    (_, dN_dxi) = fcn(points)
    nodes = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    test_polys = [(lambda x, y: 1.0, lambda x, y: (0.0, 0.0)), (lambda x, y: x, lambda x, y: (1.0, 0.0)), (lambda x, y: y, lambda x, y: (0.0, 1.0)), (lambda x, y: x * y, lambda x, y: (y, x)), (lambda x, y: x * x, lambda x, y: (2 * x, 0.0)), (lambda x, y: y * y, lambda x, y: (0.0, 2 * y))]
    for (poly, grad_poly) in test_polys:
        nodal_vals = np.array([poly(n[0], n[1]) for n in nodes]).reshape(6, 1)
        exact_grads = np.array([grad_poly(p[0], p[1]) for p in points])
        interpolated_grads = dN_dxi.reshape(-1, 6, 2) @ nodal_vals
        interpolated_grads = interpolated_grads.reshape(-1, 2)
        assert np.allclose(interpolated_grads, exact_grads, atol=1e-12)