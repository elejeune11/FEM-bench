def test_tri6_shape_functions_and_derivatives_input_errors(fcn: Callable):
    """Test that invalid inputs raise appropriate ValueError exceptions."""
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
    """Test that shape functions sum to 1 everywhere."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3]])
    (N, _) = fcn(sample_points)
    sums = np.sum(N, axis=1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn: Callable):
    """Test that shape function derivatives sum to 0."""
    sample_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5], [1 / 3, 1 / 3]])
    (_, dN) = fcn(sample_points)
    derivative_sums = np.sum(dN, axis=1)
    np.testing.assert_allclose(derivative_sums, 0.0, rtol=1e-14, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn: Callable):
    """Test that shape functions satisfy Kronecker delta property at nodes."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    N = N.squeeze()
    np.testing.assert_allclose(N, np.eye(6), rtol=1e-14, atol=1e-14)

def test_value_completeness_tri6(fcn: Callable):
    """Test that shape functions exactly reproduce quadratic polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def quad_poly(x, y):
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * x + 5.0 * x * y + 6.0 * y * y
    nodal_values = quad_poly(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(sample_points)
    interpolated = N.squeeze() @ nodal_values
    exact = quad_poly(sample_points[:, 0], sample_points[:, 1])
    np.testing.assert_allclose(interpolated, exact, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_tri6(fcn: Callable):
    """Test that shape functions exactly reproduce gradients of quadratic polynomials."""
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    sample_points = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def quad_poly(x, y):
        return 1.0 + 2.0 * x + 3.0 * y + 4.0 * x * x + 5.0 * x * y + 6.0 * y * y

    def quad_poly_grad_x(x, y):
        return 2.0 + 8.0 * x + 5.0 * y

    def quad_poly_grad_y(x, y):
        return 3.0 + 5.0 * x + 12.0 * y
    nodal_values = quad_poly(nodes[:, 0], nodes[:, 1])
    (_, dN) = fcn(sample_points)
    grad_x = dN[:, :, 0] @ nodal_values
    grad_y = dN[:, :, 1] @ nodal_values
    exact_grad_x = quad_poly_grad_x(sample_points[:, 0], sample_points[:, 1])
    exact_grad_y = quad_poly_grad_y(sample_points[:, 0], sample_points[:, 1])
    np.testing.assert_allclose(grad_x, exact_grad_x, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(grad_y, exact_grad_y, rtol=1e-14, atol=1e-14)