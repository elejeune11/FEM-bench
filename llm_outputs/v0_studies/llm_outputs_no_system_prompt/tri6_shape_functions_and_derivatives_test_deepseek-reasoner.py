def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn([0.2, 0.3])
    with pytest.raises(ValueError):
        fcn((0.2, 0.3))
    with pytest.raises(ValueError):
        fcn(np.array([0.2]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2]]))
    with pytest.raises(ValueError):
        fcn(np.array([0.2, 0.3, 0.4]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, 0.3, 0.4]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.3]))
    with pytest.raises(ValueError):
        fcn(np.array([[0.2, np.nan]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.4, 0.1], [0.1, 0.1]])
    (N, _) = fcn(sample_points)
    sum_N = np.sum(N, axis=1)
    assert np.allclose(sum_N, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5], [0.2, 0.3], [0.4, 0.1], [0.1, 0.1]])
    (_, dN_dxi) = fcn(sample_points)
    sum_dN = np.sum(dN_dxi, axis=1)
    assert np.allclose(sum_dN, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    identity_matrix = np.eye(6)
    assert np.allclose(N.reshape(6, 6), identity_matrix, atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3], [0.25, 0.25]])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    linear_polys = [lambda xi, eta: 1.0, lambda xi, eta: xi, lambda xi, eta: eta, lambda xi, eta: 1.0 - xi - eta]
    quadratic_polys = [lambda xi, eta: xi ** 2, lambda xi, eta: eta ** 2, lambda xi, eta: xi * eta, lambda xi, eta: xi * (1.0 - xi - eta), lambda xi, eta: eta * (1.0 - xi - eta)]
    (N, _) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    for poly in linear_polys + quadratic_polys:
        u_nodal = np.array([poly(node[0], node[1]) for node in nodes]).reshape(6, 1)
        u_interp = np.dot(N, u_nodal).flatten()
        u_exact = np.array([poly(pt[0], pt[1]) for pt in sample_points])
        assert np.allclose(u_interp, u_exact, atol=1e-12)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    sample_points = np.array([[0.2, 0.3], [0.4, 0.1], [0.1, 0.4], [0.3, 0.3], [0.25, 0.25]])
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    poly_grad_pairs = [(lambda xi, eta: 1.0, lambda xi, eta: (0.0, 0.0)), (lambda xi, eta: xi, lambda xi, eta: (1.0, 0.0)), (lambda xi, eta: eta, lambda xi, eta: (0.0, 1.0)), (lambda xi, eta: 1.0 - xi - eta, lambda xi, eta: (-1.0, -1.0)), (lambda xi, eta: xi ** 2, lambda xi, eta: (2.0 * xi, 0.0)), (lambda xi, eta: eta ** 2, lambda xi, eta: (0.0, 2.0 * eta)), (lambda xi, eta: xi * eta, lambda xi, eta: (eta, xi)), (lambda xi, eta: xi * (1.0 - xi - eta), lambda xi, eta: (1.0 - 2.0 * xi - eta, -xi)), (lambda xi, eta: eta * (1.0 - xi - eta), lambda xi, eta: (-eta, 1.0 - xi - 2.0 * eta))]
    (N, dN_dxi) = fcn(sample_points)
    (N_nodes, _) = fcn(nodes)
    for (poly, grad_poly) in poly_grad_pairs:
        u_nodal = np.array([poly(node[0], node[1]) for node in nodes]).reshape(6, 1)
        grad_interp = np.dot(dN_dxi.transpose(0, 2, 1), u_nodal).reshape(-1, 2)
        grad_exact = np.array([grad_poly(pt[0], pt[1]) for pt in sample_points])
        assert np.allclose(grad_interp, grad_exact, atol=1e-12)