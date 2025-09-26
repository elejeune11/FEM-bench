def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
    """The quad8 vectorized evaluator requires a NumPy array input of shape (2,) or (n, 2)
    with finite values. Invalid inputs must raise ValueError."""
    with pytest.raises(ValueError):
        fcn([1.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([1.0]))
    with pytest.raises(ValueError):
        fcn(np.array([1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, np.nan]]))

def test_partition_of_unity_quad8(fcn):
    """Shape functions on a quadrilateral must satisfy the partition of unity."""
    xi_samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (N, _) = fcn(xi_samples)
    sum_N = np.sum(N, axis=1)
    np.testing.assert_allclose(sum_N, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Partition of unity implies sum of gradients equals zero."""
    xi_samples = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    (_, dN) = fcn(xi_samples)
    sum_dN = np.sum(dN, axis=1)
    np.testing.assert_allclose(sum_dN, 0.0, rtol=1e-14, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """Shape functions satisfy Kronecker delta property at nodes."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    (N, _) = fcn(nodes)
    N = N.reshape(-1, 8)
    np.testing.assert_allclose(N, np.eye(8), rtol=1e-14, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """Verify exact reproduction of quadratic polynomials."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.array([[0.25, -0.75], [-0.5, 0.5], [0.3, 0.3]])

    def exact_values(x, y):
        return np.array([np.ones_like(x), x, y, x * x, x * y, y * y]).T
    nodal_vals = exact_values(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(xi)
    for i in range(6):
        interpolated = np.sum(N * nodal_vals[:, i].reshape(1, -1, 1), axis=1)
        exact = exact_values(xi[:, 0], xi[:, 1])[:, i:i + 1]
        np.testing.assert_allclose(interpolated, exact, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_quad8(fcn):
    """Verify exact reproduction of gradients of quadratic polynomials."""
    nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    xi = np.array([[0.25, -0.75], [-0.5, 0.5], [0.3, 0.3]])

    def exact_values(x, y):
        return np.array([np.ones_like(x), x, y, x * x, x * y, y * y]).T

    def exact_gradients(x, y):
        return np.array([[np.zeros_like(x), np.zeros_like(x)], [np.ones_like(x), np.zeros_like(x)], [np.zeros_like(x), np.ones_like(x)], [2 * x, np.zeros_like(x)], [y, x], [np.zeros_like(x), 2 * y]]).transpose(1, 0, 2)
    nodal_vals = exact_values(nodes[:, 0], nodes[:, 1])
    (_, dN) = fcn(xi)
    for i in range(6):
        interpolated_grad = np.sum(dN * nodal_vals[:, i].reshape(1, -1, 1), axis=1)
        exact_grad = exact_gradients(xi[:, 0], xi[:, 1])[:, i, :]
        np.testing.assert_allclose(interpolated_grad, exact_grad, rtol=1e-14, atol=1e-14)