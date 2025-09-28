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
    sums = np.sum(N, axis=1)
    assert np.allclose(sums, 1.0, rtol=1e-14, atol=1e-14)

def test_derivative_partition_of_unity_quad8(fcn):
    """Test that shape function derivatives sum to 0."""
    import numpy as np
    xi_samples = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]])
    (_, dN) = fcn(xi_samples)
    deriv_sums = np.sum(dN, axis=1)
    assert np.allclose(deriv_sums, 0.0, rtol=1e-14, atol=1e-14)

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
    xi_samples = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def f1(x, y):
        return 2 * x + 3 * y + 1

    def f2(x, y):
        return x ** 2 + 2 * x * y + 3 * y ** 2 + 4 * x + 5 * y + 6
    for f in [f1, f2]:
        nodal_vals = f(nodes[:, 0], nodes[:, 1])
        (N, _) = fcn(xi_samples)
        interp_vals = np.sum(N * nodal_vals, axis=1)
        exact_vals = f(xi_samples[:, 0], xi_samples[:, 1])
        assert np.allclose(interp_vals, exact_vals, rtol=1e-14, atol=1e-14)

def test_gradient_completeness_quad8(fcn):
    """Test that shape functions exactly reproduce gradients of quadratic polynomials."""
    import numpy as np
    nodes = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]])
    xi_samples = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5], [0, -0.5], [0.5, 0], [0, 0.5], [-0.5, 0], [0, 0]])

    def f1(x, y):
        return 2 * x + 3 * y + 1

    def df1_dx(x, y):
        return 2 + 0 * x

    def df1_dy(x, y):
        return 3 + 0 * x

    def f2(x, y):
        return x ** 2 + 2 * x * y + 3 * y ** 2 + 4 * x + 5 * y + 6

    def df2_dx(x, y):
        return 2 * x + 2 * y + 4

    def df2_dy(x, y):
        return 2 * x + 6 * y + 5
    for (f, fx, fy) in [(f1, df1_dx, df1_dy), (f2, df2_dx, df2_dy)]:
        nodal_vals = f(nodes[:, 0], nodes[:, 1])
        (_, dN) = fcn(xi_samples)
        interp_dx = np.sum(dN[:, :, 0] * nodal_vals, axis=1)
        interp_dy = np.sum(dN[:, :, 1] * nodal_vals, axis=1)
        exact_dx = fx(xi_samples[:, 0], xi_samples[:, 1])
        exact_dy = fy(xi_samples[:, 0], xi_samples[:, 1])
        assert np.allclose(interp_dx, exact_dx, rtol=1e-14, atol=1e-14)
        assert np.allclose(interp_dy, exact_dy, rtol=1e-14, atol=1e-14)