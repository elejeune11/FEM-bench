def test_quad8_shape_functions_and_derivatives_input_errors(fcn):
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
        fcn(np.array([[0, 0, 0]]))
    with pytest.raises(ValueError):
        fcn(np.array([np.nan, 0]))
    with pytest.raises(ValueError):
        fcn(np.array([np.inf, 0]))

def test_partition_of_unity_quad8(fcn):
    """Test that shape functions sum to 1 at sample points."""
    import numpy as np
    xi_samples = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (N, _) = fcn(xi_samples)
    N_sum = np.sum(N, axis=1)
    assert np.allclose(N_sum, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Test that shape function derivatives sum to 0 at sample points."""
    import numpy as np
    xi_samples = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN) = fcn(xi_samples)
    dN_sum = np.sum(dN, axis=1)
    assert np.allclose(dN_sum, 0.0, rtol=1e-14, atol=1e-14)

def test_kronecker_delta_at_nodes_quad8(fcn):
    """Test that shape functions satisfy Kronecker delta property at nodes."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    (N, _) = fcn(nodes)
    N = N.reshape(-1, 8)
    assert np.allclose(N, np.eye(8), rtol=1e-14, atol=1e-14)

def test_value_completeness_quad8(fcn):
    """Test that shape functions exactly reproduce quadratic polynomials."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def exact_poly(x, y):
        return 1 + 2 * x + 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2
    nodal_vals = exact_poly(nodes[:, 0], nodes[:, 1])
    (N, _) = fcn(xi)
    interp_vals = np.sum(N * nodal_vals, axis=1)
    exact_vals = exact_poly(xi[:, 0], xi[:, 1])
    assert np.allclose(interp_vals, exact_vals, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_quad8(fcn):
    """Test that shape functions exactly reproduce quadratic polynomial gradients."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def exact_poly(x, y):
        return 1 + 2 * x + 3 * y + 4 * x * y + 5 * x ** 2 + 6 * y ** 2

    def exact_grad_x(x, y):
        return 2 + 4 * y + 10 * x

    def exact_grad_y(x, y):
        return 3 + 4 * x + 12 * y
    nodal_vals = exact_poly(nodes[:, 0], nodes[:, 1])
    (_, dN) = fcn(xi)
    grad_x = np.sum(dN[:, :, 0] * nodal_vals, axis=1)
    grad_y = np.sum(dN[:, :, 1] * nodal_vals, axis=1)
    exact_dx = exact_grad_x(xi[:, 0], xi[:, 1])
    exact_dy = exact_grad_y(xi[:, 0], xi[:, 1])
    assert np.allclose(grad_x, exact_dx, rtol=1e-14, atol=1e-14)
    assert np.allclose(grad_y, exact_dy, rtol=1e-14, atol=1e-14)