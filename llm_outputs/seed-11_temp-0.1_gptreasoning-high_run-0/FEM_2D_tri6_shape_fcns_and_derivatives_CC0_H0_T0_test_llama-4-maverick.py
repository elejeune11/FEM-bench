def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """D2_nn6_tri_with_grad_vec enforces that inputs are NumPy arrays of shape (2,) or (n,2) with finite values. Invalid inputs should raise ValueError."""
    with pytest.raises(ValueError):
        fcn('not a numpy array')
    with pytest.raises(ValueError):
        fcn(np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, 2], [3, 4], [5, 6, 7]]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, np.nan], [3, 4]]))
    with pytest.raises(ValueError):
        fcn(np.array([[1, np.inf], [3, 4]]))

def test_partition_of_unity_tri6(fcn):
    """Shape functions on a triangle must satisfy the partition of unity: ∑_{i=1}^6 N_i(ξ,η) = 1 for all (ξ,η) in the reference triangle."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(sample_points)
    assert np.allclose(np.sum(N, axis=1), np.ones((sample_points.shape[0], 1)))

def test_derivative_partition_of_unity_tri6(fcn):
    """Partition of unity implies ∑_i ∇N_i = 0, ensuring constant fields have zero gradient after interpolation."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (_, dN_dxi) = fcn(sample_points)
    assert np.allclose(np.sum(dN_dxi, axis=1), np.zeros((sample_points.shape[0], 2)))

def test_kronecker_delta_at_nodes_tri6(fcn):
    """For Lagrange P2 triangles, N_i equals 1 at its own node and 0 at all others."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    assert np.allclose(N.squeeze(), np.eye(6))

def test_value_completeness_tri6(fcn):
    """Check that quadratic (P2) triangle shape functions exactly reproduce degree-1 and degree-2 polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.random.rand(100, 2)
    sample_points = sample_points / (sample_points.sum(axis=1, keepdims=True) + 1e-12)
    for degree in [1, 2]:
        for coeff in np.random.rand(degree + 1, degree + 1):
            u_exact = lambda x: coeff * x[:, 0] ** degree * x[:, 1] ** (degree - np.arange(degree + 1))
            u_nodes = u_exact(nodes)
            (N, _) = fcn(sample_points)
            u_interp = np.sum(N * u_nodes[:, None, None], axis=1).squeeze()
            assert np.allclose(u_interp, u_exact(sample_points), atol=1e-10)

def test_gradient_completeness_tri6(fcn):
    """Check that P2 triangle shape functions reproduce the exact gradient for degree-1 and degree-2 polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.random.rand(100, 2)
    sample_points = sample_points / (sample_points.sum(axis=1, keepdims=True) + 1e-12)
    for degree in [1, 2]:
        for coeff in np.random.rand(degree + 1, degree + 1):
            u_exact = lambda x: coeff * x[:, 0] ** degree * x[:, 1] ** (degree - np.arange(degree + 1))
            du_exact_dx = lambda x: coeff * degree * x[:, 0] ** (degree - 1) * x[:, 1] ** (degree - np.arange(degree + 1))
            du_exact_dy = lambda x: coeff * x[:, 0] ** degree * (degree - np.arange(degree + 1)) * x[:, 1] ** (degree - np.arange(degree + 1) - 1)
            u_nodes = u_exact(nodes)
            (_, dN_dxi) = fcn(sample_points)
            du_interp_dx = np.sum(dN_dxi[:, :, 0] * u_nodes[:, None], axis=1).squeeze()
            du_interp_dy = np.sum(dN_dxi[:, :, 1] * u_nodes[:, None], axis=1).squeeze()
            assert np.allclose(du_interp_dx, du_exact_dx(sample_points), atol=1e-10)
            assert np.allclose(du_interp_dy, du_exact_dy(sample_points), atol=1e-10)