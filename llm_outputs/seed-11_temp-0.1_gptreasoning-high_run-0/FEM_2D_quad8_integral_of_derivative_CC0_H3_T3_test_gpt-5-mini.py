def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case."""
    import numpy as np
    nodes_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = nodes_ref.copy()

    def u(x, y):
        return 1.5 + 0.7 * x - 1.2 * y + 2.3 * x * x * y - 1.7 * x * y * y + 0.9 * x ** 3 - 0.4 * y ** 3
    node_values = u(node_coords[:, 0], node_coords[:, 1])
    res4 = fcn(node_coords, node_values, 4)
    res9 = fcn(node_coords, node_values, 9)
    assert res4.shape == (2,)
    assert res9.shape == (2,)
    assert np.allclose(res4, res9, rtol=1e-08, atol=1e-09)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """Analytic check with an affine geometric map and a linear field.
    Map the reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c.
    For the linear scalar field u(x, y) = α + βx + γy.
    Test to make sure the function matches the correct analytical solution."""
    import numpy as np
    nodes_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    A = np.array([[2.0, 0.5], [-0.3, 1.7]], dtype=float)
    c = np.array([0.7, -1.2], dtype=float)
    node_coords = (A @ nodes_ref.T).T + c
    alpha = 1.234
    beta = 0.75
    gamma = -0.6
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    res = fcn(node_coords, node_values, 4)
    area = 4.0 * float(np.linalg.det(A))
    expected = np.array([beta * area, gamma * area], dtype=float)
    assert res.shape == (2,)
    assert np.allclose(res, expected, rtol=1e-08, atol=1e-09)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Select a geometry that is intentionally non-affine. With properly selected fixed,
    asymmetric nodal values, the integral ∫_Ω (∇u) dΩ should depend on the
    quadrature order. The test confirms that results from a 3×3 rule differ from those 
    of 1×1 or 2×2, verifying that higher-order integration is sensitive to nonlinear mappings."""
    import numpy as np
    nodes_ref = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = nodes_ref.copy()
    node_coords[4] = np.array([0.0, -0.6], dtype=float)
    node_coords[5] = np.array([1.4, 0.0], dtype=float)
    node_coords[6] = np.array([0.0, 0.8], dtype=float)
    node_coords[7] = np.array([-1.2, 0.0], dtype=float)
    node_values = np.array([1.0, -0.5, 2.3, -1.1, 0.9, -2.4, 3.5, 1.7], dtype=float)
    res1 = fcn(node_coords, node_values, 1)
    res4 = fcn(node_coords, node_values, 4)
    res9 = fcn(node_coords, node_values, 9)
    assert res9.shape == (2,)
    assert not np.allclose(res9, res4, rtol=1e-06, atol=1e-09)
    assert not np.allclose(res9, res1, rtol=1e-06, atol=1e-09)