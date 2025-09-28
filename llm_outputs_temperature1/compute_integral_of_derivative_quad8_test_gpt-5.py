def test_integral_of_derivative_quad8_identity_cubic(fcn):
    """
    Analytic check with identity mapping (reference element = physical element).
    When x ≡ ξ and y ≡ η, the Jacobian is J = I with det(J) = 1, so the routine
    integrates the physical gradient directly over Q = [-1, 1] × [-1, 1].
    Using u(x, y) = x^3 + y^3 gives ∇u = [3x^2, 3y^2] and the exact integrals on Q are [4, 4].
    A 2×2 Gauss rule (num_gauss_pts=4) is exact for this case; the test checks we recover [4, 4].
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = x ** 3 + y ** 3
    res = np.asarray(fcn(node_coords, node_values, 4)).ravel()
    expected = np.array([4.0, 4.0])
    assert res.shape == (2,)
    assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_affine_linear_field(fcn):
    """
    Analytic check with an affine geometric map and a linear field. We map the
    reference square Q = [-1, 1] × [-1, 1] by [x, y]^T = A[ξ, η]^T + c. By placing
    the eight node quad mid-edge nodes at the arithmetic midpoints of the mapped corners, the
    isoparametric geometry is exactly affine, so the Jacobian is constant (J = A,
    det(J) = det(A)). For the linear scalar field u(x, y) = α + βx + γy, the physical
    gradient is constant ∇u = [β, γ], hence ∫_Ω ∇u dΩ = [β, γ] · Area(Ω). The area follows
    from the mapping: Area(Ω) = ∫_Q det(J) dQ = det(A) · Area(Q) = 4 · |det(A)|, so
    the exact result is [β, γ] · (4 · |det(A)|). Test to make sure the function matches
    this analytical solution.
    """
    A = np.array([[2.0, 0.5], [-0.3, 1.5]], dtype=float)
    c = np.array([0.7, -1.2], dtype=float)
    ref_nodes = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=float)
    corners = ref_nodes[:4]
    mapped_corners = corners @ A.T + c
    (n1, n2, n3, n4) = mapped_corners
    mapped_mids = np.array([0.5 * (n1 + n2), 0.5 * (n2 + n3), 0.5 * (n3 + n4), 0.5 * (n4 + n1)], dtype=float)
    node_coords = np.vstack([mapped_corners, mapped_mids])
    (alpha, beta, gamma) = (0.4, 1.2, -0.8)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    node_values = alpha + beta * x + gamma * y
    res = np.asarray(fcn(node_coords, node_values, 4)).ravel()
    expected_area = 4.0 * float(np.linalg.det(A))
    expected = np.array([beta, gamma]) * expected_area
    assert res.shape == (2,)
    assert np.allclose(res, expected, rtol=1e-12, atol=1e-12)

def test_integral_of_derivative_quad8_order_check_asymmetric_curved(fcn):
    """
    Test quadrature-order sensitivity on a deliberately curved, asymmetric mapping.
    One approach is to keep the four corners on the reference square but displace the mid-edge
    nodes asymmetrically, inducing a non-affine geometry (spatially varying J).
    With fixed, non-symmetric nodal values, the FE integrand becomes high-order in (ξ, η), 
    so a 3×3 rule should not coincide with 2×2 or 1×1.
    The test asserts that increasing the rule to 3×3 changes the result.
    """
    node_coords = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [0.2, -1.0], [1.0, 0.3], [-0.25, 0.9], [-1.0, -0.2]], dtype=float)
    node_values = np.array([0.3, -1.1, 2.2, 0.7, -0.4, 1.5, -2.3, 0.9], dtype=float)
    res_1 = np.asarray(fcn(node_coords, node_values, 1)).ravel()
    res_4 = np.asarray(fcn(node_coords, node_values, 4)).ravel()
    res_9 = np.asarray(fcn(node_coords, node_values, 9)).ravel()
    assert res_1.shape == (2,)
    assert res_4.shape == (2,)
    assert res_9.shape == (2,)
    assert not np.allclose(res_9, res_4, rtol=1e-09, atol=1e-09)
    assert not np.allclose(res_9, res_1, rtol=1e-09, atol=1e-09)