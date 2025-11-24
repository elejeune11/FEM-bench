def test_tri6_shape_functions_and_derivatives_input_errors(fcn):
    """Test that invalid inputs raise ValueError."""
    import numpy as np
    import pytest
    with pytest.raises(ValueError):
        fcn([0, 0])
    with pytest.raises(ValueError):
        fcn(np.array([0]))
    with pytest.raises(ValueError):
        fcn(np.array([0, 0, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0]))

def test_partition_of_unity_tri6(fcn):
    """Test that shape functions sum to 1 everywhere."""
    import numpy as np
    xi = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0.25, 0.25], [0.1, 0.8]])
    (N, _) = fcn(xi)
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-14)

def test_derivative_partition_of_unity_tri6(fcn):
    """Test that shape function derivatives sum to 0."""
    import numpy as np
    xi = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5], [0.25, 0.25]])
    (_, dN) = fcn(xi)
    derivative_sums = np.sum(dN, axis=1)
    assert np.allclose(derivative_sums, 0.0, atol=1e-14)

def test_kronecker_delta_at_nodes_tri6(fcn):
    """Test that shape functions equal 1 at their own node and 0 at others."""
    import numpy as np
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    (N, _) = fcn(nodes)
    N = N.reshape(6, 6)
    assert np.allclose(N, np.eye(6), atol=1e-14)

def test_value_completeness_tri6(fcn):
    """Test exact reproduction of quadratic polynomials."""
    import numpy as np
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    xi = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def p(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2
    nodal_values = p(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(xi)
    p_interp = N @ nodal_values.reshape(-1, 1)
    p_exact = p(xi[:, 0], xi[:, 1])
    assert np.allclose(p_interp.flatten(), p_exact, atol=1e-14)

def test_gradient_completeness_tri6(fcn):
    """Test exact reproduction of gradients of quadratic polynomials."""
    import numpy as np
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    xi = np.array([[0.25, 0.25], [0.1, 0.7], [0.6, 0.2]])

    def p(x, y):
        return 1 + 2 * x + 3 * y + 4 * x ** 2 + 5 * x * y + 6 * y ** 2

    def dp_dx(x, y):
        return 2 + 8 * x + 5 * y

    def dp_dy(x, y):
        return 3 + 5 * x + 12 * y
    nodal_values = p(nodes[:, 0], nodes[:, 1])
    (_, dN) = fcn(xi)
    grad_x = np.sum(dN[:, :, 0] * nodal_values, axis=1)
    grad_y = np.sum(dN[:, :, 1] * nodal_values, axis=1)
    grad_x_exact = dp_dx(xi[:, 0], xi[:, 1])
    grad_y_exact = dp_dy(xi[:, 0], xi[:, 1])
    assert np.allclose(grad_x, grad_x_exact, atol=1e-14)
    assert np.allclose(grad_y, grad_y_exact, atol=1e-14)