def test_tri6_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """Test that invalid inputs raise ValueError."""
    with pytest.raises(ValueError):
        fcn([1.0, 0.0])
    with pytest.raises(ValueError):
        fcn(np.array([1.0]))
    with pytest.raises(ValueError):
        fcn(np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError):
        fcn(np.array([[np.inf, 0.0]]))

def test_partition_of_unity_tri6(fcn: Callable):
    """Test partition of unity property."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn: Callable):
    """Test that shape function derivatives sum to zero."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3]])
    (_, dN) = fcn(sample_points)
    derivative_sums = np.sum(dN, axis=1)
    assert np.allclose(derivative_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn: Callable):
    """Test Kronecker delta property at nodes."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    N = N.squeeze()
    assert np.allclose(N, np.eye(6), atol=1e-14)

def test_value_completeness_tri6(fcn: Callable):
    """Test exact reproduction of quadratic polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def p(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2
    nodal_values = p(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(sample_points)
    interpolated = N.squeeze() @ nodal_values
    exact = p(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(interpolated, exact, atol=1e-14)

def test_gradient_completeness_tri6(fcn: Callable):
    """Test exact reproduction of gradients of quadratic polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def p(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2

    def dp_dx(x, y):
        return 2 + 8 * x + 5 * y

    def dp_dy(x, y):
        return 3 + 5 * x + 12 * y
    nodal_values = p(nodes[:, 0], nodes[:, 1])
    (_, dN) = fcn(sample_points)
    grad_x = dN[:, :, 0] @ nodal_values
    grad_y = dN[:, :, 1] @ nodal_values
    exact_dx = dp_dx(sample_points[:, 0], sample_points[:, 1])
    exact_dy = dp_dy(sample_points[:, 0], sample_points[:, 1])
    assert np.allclose(grad_x, exact_dx, atol=1e-14)
    assert np.allclose(grad_y, exact_dy, atol=1e-14)