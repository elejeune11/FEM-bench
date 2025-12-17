def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Validate ∫_Ω (∇u) dΩ for a cubic field on an identity-mapped Q8 element.
    Use u(x, y) = x^2*y + x*y^2 on Ω = [-1, 1]^2.
    For this field, ∫_Ω ∇u dΩ = [4/3, 4/3].
    A 2×2 Gauss–Legendre rule (num_gauss_pts = 4) is exact for this case.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)

    def u(x, y):
        return x * x * y + x * y * y
    node_values = np.array([u(x, y) for x, y in node_coords], dtype=float)
    res = fcn(node_coords, node_values, 4)
    expected = np.array([4.0 / 3.0, 4.0 / 3.0], dtype=float)
    assert isinstance(res, np.ndarray)
    assert res.shape == (2,)
    assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field.
    Map Q = [-1, 1]^2 by [x, y]^T = A[ξ, η]^T + c with det(A) > 0.
    For u(x, y) = α + βx + γy, ∫_Ω ∇u dΩ = [β, γ] * area(Ω) with area = 4 * det(A).
    Verify consistency for num_gauss_pts in {1, 4, 9}.
    """
    A = np.array([[2.0, 0.5], [0.3, 1.5]], dtype=float)
    c = np.array([0.2, -1.3], dtype=float)
    detA = np.linalg.det(A)
    assert detA > 0.0
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    node_coords = ref_nodes @ A.T + c
    alpha, beta, gamma = (1.2, -0.7, 2.3)
    node_values = alpha + beta * node_coords[:, 0] + gamma * node_coords[:, 1]
    area = 4.0 * detA
    expected = np.array([beta * area, gamma * area], dtype=float)
    for npts in (1, 4, 9):
        res = fcn(node_coords, node_values, npts)
        assert isinstance(res, np.ndarray)
        assert res.shape == (2,)
        assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Check quadrature-order sensitivity on a curved, asymmetric Q8 mapping.
    Use a non-affine geometry and asymmetric nodal values. Confirm that 3×3
    integration differs from 1×1, and that 3×3 is at least as close to 2×2
    as to 1×1, demonstrating higher-order integration sensitivity.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -0.8], [1.1, 1.0], [-1.0, 1.2], [0.1, -1.2], [1.3, 0.1], [0.0, 1.3], [-1.2, 0.3]], dtype=float)
    node_values = np.array([0.5, -1.0, 2.0, -0.3, 1.2, -0.7, 0.8, -1.4], dtype=float)
    r1 = fcn(node_coords, node_values, 1)
    r4 = fcn(node_coords, node_values, 4)
    r9 = fcn(node_coords, node_values, 9)
    d1 = np.linalg.norm(r9 - r1)
    d4 = np.linalg.norm(r9 - r4)
    assert d1 > 1e-10
    assert d4 <= d1