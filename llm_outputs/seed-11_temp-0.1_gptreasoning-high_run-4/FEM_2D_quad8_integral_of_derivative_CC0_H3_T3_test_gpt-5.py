def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    Use u(x, y) = x*y*(x - y). On Ω = [-1, 1] × [-1, 1], the exact integral is
    [∫∂u/∂x, ∫∂u/∂y] = [-4/3, 4/3]. A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) should be exact.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

    def u(x, y):
        return x * y * (x - y)
    node_values = np.array([u(x, y) for x, y in node_coords])
    result = fcn(node_coords, node_values, 4)
    expected = np.array([-4.0 / 3.0, 4.0 / 3.0])
    assert result.shape == (2,)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map Q = [-1, 1]^2 by [x, y]^T = A[ξ, η]^T + c with det(A) > 0.
    For u(x, y) = α + βx + γy, ∫_Ω ∇u dΩ = [β, γ] * area(Ω) = [β, γ] * (4 * det(A)).
    """
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    A = np.array([[2.0, 0.3], [-0.4, 1.7]])
    c = np.array([0.7, -0.9])
    detA = np.linalg.det(A)
    assert detA > 0
    node_coords = ref_nodes @ A.T + c
    alpha, beta, gamma = (1.2, -0.8, 2.3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    result = fcn(node_coords, node_values, 1)
    expected = np.array([beta, gamma]) * (4.0 * detA)
    assert result.shape == (2,)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Use a non-affine geometry (curved edges) and asymmetric nodal values.
    Confirm that results with a 3×3 rule differ from those with 1×1 and 2×2 rules.
    """
    node_coords = np.array([[-1.0, -1.0], [2.0, -0.8], [2.2, 2.1], [-1.2, 2.0], [0.3, -1.2], [2.4, 0.6], [0.1, 2.4], [-1.4, 0.8]])
    node_values = np.array([1.0, -0.5, 0.2, 1.1, -0.7, 0.9, -1.3, 0.3])
    res_1 = fcn(node_coords, node_values, 1)
    res_4 = fcn(node_coords, node_values, 4)
    res_9 = fcn(node_coords, node_values, 9)
    assert np.all(np.isfinite(res_1))
    assert np.all(np.isfinite(res_4))
    assert np.all(np.isfinite(res_9))
    assert not np.allclose(res_9, res_1, rtol=1e-06, atol=1e-08)
    assert not np.allclose(res_9, res_4, rtol=1e-06, atol=1e-08)