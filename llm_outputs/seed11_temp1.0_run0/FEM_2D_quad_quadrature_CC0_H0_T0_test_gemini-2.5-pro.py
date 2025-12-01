def test_quad_quadrature_2D_invalid_inputs(fcn: Callable):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 3, 5, 8, 10, -1, 16]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_quad_quadrature_2D_basics(fcn: Callable):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(points >= -1.0)
        assert np.all(points <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn: Callable):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    The tensor-product 1-point rule places a single node at the center (0,0) with
    weight 4. This integrates exactly any polynomial that is at most degree 1
    in each variable, i.e. constants and linear terms in x or y.
    For higher-degree terms, the rule is no longer guaranteed to be exact.
    Exactness assertions for monomials of degree ≤ 1 should pass.
    Non-exactness assertions for quadratics should fail the exactness check
    (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)
    (xi, eta) = (points[:, 0], points[:, 1])

    def integrate(f):
        return np.sum(weights * f(xi, eta))
    exact_cases = [(lambda x, y: 1.0 + 0 * x, 4.0), (lambda x, y: x, 0.0), (lambda x, y: y, 0.0), (lambda x, y: x * y, 0.0), (lambda x, y: 3 * x - 2 * y + 5, 20.0)]
    for (f, exact_integral) in exact_cases:
        assert integrate(f) == pytest.approx(exact_integral)
    non_exact_cases = [(lambda x, y: x ** 2, 4.0 / 3.0), (lambda x, y: y ** 2, 4.0 / 3.0), (lambda x, y: x ** 2 * y ** 2, 4.0 / 9.0)]
    for (f, exact_integral) in non_exact_cases:
        assert integrate(f) != pytest.approx(exact_integral)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn: Callable):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
    Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
    any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
    in x and y).
    Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(4)
    (xi, eta) = (points[:, 0], points[:, 1])

    def integrate(f):
        return np.sum(weights * f(xi, eta))

    def exact_integral(p, q):
        integral_x = 2.0 / (p + 1.0) if p % 2 == 0 else 0.0
        integral_y = 2.0 / (q + 1.0) if q % 2 == 0 else 0.0
        return integral_x * integral_y
    for p in range(4):
        for q in range(4):
            f = lambda x, y, p=p, q=q: x ** p * y ** q
            expected = exact_integral(p, q)
            assert integrate(f) == pytest.approx(expected)
    non_exact_powers = [(4, 0), (0, 4), (2, 4), (4, 2), (4, 4)]
    for (p, q) in non_exact_powers:
        f = lambda x, y, p=p, q=q: x ** p * y ** q
        expected = exact_integral(p, q)
        assert integrate(f) != pytest.approx(expected)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn: Callable):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
    Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
    for any polynomial where the degree in each variable is ≤ 5.
    The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.
    Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    (points, weights) = fcn(9)
    (xi, eta) = (points[:, 0], points[:, 1])

    def integrate(f):
        return np.sum(weights * f(xi, eta))

    def exact_integral(p, q):
        integral_x = 2.0 / (p + 1.0) if p % 2 == 0 else 0.0
        integral_y = 2.0 / (q + 1.0) if q % 2 == 0 else 0.0
        return integral_x * integral_y
    for p in range(6):
        for q in range(6):
            f = lambda x, y, p=p, q=q: x ** p * y ** q
            expected = exact_integral(p, q)
            assert integrate(f) == pytest.approx(expected)
    non_exact_powers = [(6, 0), (0, 6), (2, 6), (6, 2), (6, 6)]
    for (p, q) in non_exact_powers:
        f = lambda x, y, p=p, q=q: x ** p * y ** q
        expected = exact_integral(p, q)
        assert integrate(f) != pytest.approx(expected)