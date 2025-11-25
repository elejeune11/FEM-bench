def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError."""
    import pytest
    invalid_values = [0, -1, 2, 3, 5, 6, 7, 8, 10, 16, 25, 100]
    for num_pts in invalid_values:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1."""
    import numpy as np
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert np.abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points[:, 0] >= -1.0)
        assert np.all(points[:, 0] <= 1.0)
        assert np.all(points[:, 1] >= -1.0)
        assert np.all(points[:, 1] <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    Background
    ----------
    • The tensor-product 1-point rule places a single node at the center (0,0) with
      weight 4. This integrates exactly any polynomial that is at most degree 1
      in each variable, i.e. constants and linear terms in x or y.
    • For higher-degree terms, the rule is no longer guaranteed to be exact.
    What this test does
    -------------------
    1. Constructs a random affine polynomial P(x,y) = a00 + a10·x + a01·y.
    2. Adds quadratic contributions (x², y², xy).
         confirming that the 1-point rule is not exact for degree ≥ 2.
    Expected outcome
    ----------------
    • Exactness assertions for monomials of degree ≤ 1 should pass.
    • Non-exactness assertions for quadratics should fail the exactness check
      (i.e. the quadrature does not reproduce the analytic integrals)."""
    import numpy as np
    (points, weights) = fcn(1)
    np.random.seed(42)
    a00 = np.random.randn()
    a10 = np.random.randn()
    a01 = np.random.randn()

    def poly1(x, y):
        return a00 + a10 * x + a01 * y
    quad_result = np.sum(weights * poly1(points[:, 0], points[:, 1]))
    exact_integral = 4.0 * a00
    assert np.abs(quad_result - exact_integral) < 1e-14
    a20 = np.random.randn()
    a02 = np.random.randn()
    a11 = np.random.randn()

    def poly2(x, y):
        return a00 + a10 * x + a01 * y + a20 * x ** 2 + a02 * y ** 2 + a11 * x * y
    quad_result2 = np.sum(weights * poly2(points[:, 0], points[:, 1]))
    exact_integral2 = 4.0 * a00 + 4.0 / 3.0 * (a20 + a02)
    assert np.abs(quad_result2 - exact_integral2) > 1e-10

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Background
    ----------
    • The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
      Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
      any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
      in x and y).
    • It is not guaranteed to integrate quartic terms (degree 4 in either variable) exactly.
    What this test does
    -------------------
    1. Builds a random polynomial P(x,y) = Σ a_ij x^i y^j for 0 ≤ i,j ≤ 3, i.e. all terms up
       to cubic in x and cubic in y. The quadrature approximation is compared to the analytic
       integral, the two results must match.
    2. Extends the polynomial with quartic terms x^4 and y^4. For these terms the 2×2 rule
       is not exact, so the quadrature result is expected to deviate from the analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    • Adding quartic terms should break exactness, and the mismatch is detected by the test."""
    import numpy as np
    (points, weights) = fcn(4)
    np.random.seed(43)
    coeffs = np.random.randn(4, 4)

    def poly_cubic(x, y):
        result = 0.0
        for i in range(4):
            for j in range(4):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * poly_cubic(points[:, 0], points[:, 1]))
    exact_integral = 0.0
    for i in range(4):
        for j in range(4):
            if i % 2 == 0 and j % 2 == 0:
                exact_integral += coeffs[i, j] * (2.0 / (i + 1)) * (2.0 / (j + 1))
    assert np.abs(quad_result - exact_integral) < 1e-14
    a40 = np.random.randn()
    a04 = np.random.randn()

    def poly_quartic(x, y):
        return poly_cubic(x, y) + a40 * x ** 4 + a04 * y ** 4
    quad_result2 = np.sum(weights * poly_quartic(points[:, 0], points[:, 1]))
    exact_integral2 = exact_integral + a40 * (2.0 / 5.0) * 2.0 + a04 * 2.0 * (2.0 / 5.0)
    assert np.abs(quad_result2 - exact_integral2) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    Background
    ----------
    • The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
      Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
      for any polynomial where the degree in each variable is ≤ 5.
    • The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    What this test does
    -------------------
    1. Constructs a random polynomial P(x,y) = Σ a_ij x^i y^j with 0 ≤ i,j ≤ 5 (up to quintic
       terms in both variables). The quadrature approximation is compared to the analytic
       integral, the two results must match within tight tolerance.
    2. Extends the polynomial with degree-6 contributions (x^6, y^6, x^4y^2). These exceed the
       rule's guaranteed exactness, so the quadrature result is expected to deviate from the
       analytic integral.
    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    • Adding degree-6 terms should break exactness, and the mismatch is detected by the test."""
    import numpy as np
    (points, weights) = fcn(9)
    np.random.seed(44)
    coeffs = np.random.randn(6, 6)

    def poly_quintic(x, y):
        result = 0.0
        for i in range(6):
            for j in range(6):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * poly_quintic(points[:, 0], points[:, 1]))
    exact_integral = 0.0
    for i in range(6):
        for j in range(6):
            if i % 2 == 0 and j % 2 == 0:
                exact_integral += coeffs[i, j] * (2.0 / (i + 1)) * (2.0 / (j + 1))
    assert np.abs(quad_result - exact_integral) < 1e-13
    a60 = np.random.randn()
    a06 = np.random.randn()
    a42 = np.random.randn()

    def poly_sextic(x, y):
        return poly_quintic(x, y) + a60 * x ** 6 + a06 * y ** 6 + a42 * x ** 4 * y ** 2
    quad_result2 = np.sum(weights * poly_sextic(points[:, 0], points[:, 1]))
    exact_integral2 = exact_integral + a60 * (2.0 / 7.0) * 2.0 + a06 * 2.0 * (2.0 / 7.0) + a42 * (2.0 / 5.0) * (2.0 / 3.0)
    assert np.abs(quad_result2 - exact_integral2) > 1e-10