def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 5, 6, -1, 10, 100]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    for num_pts in [1, 3, 4]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts} points'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts} points'
        assert points.dtype == np.float64, f'Points dtype should be float64'
        assert weights.dtype == np.float64, f'Weights dtype should be float64'
        weight_sum = np.sum(weights)
        assert np.isclose(weight_sum, 0.5), f'Weights sum to {weight_sum}, expected 0.5'
        x = points[:, 0]
        y = points[:, 1]
        assert np.all(x >= -1e-14), f'Some x coordinates are negative'
        assert np.all(y >= -1e-14), f'Some y coordinates are negative'
        assert np.all(x + y <= 1.0 + 1e-14), f'Some points violate x + y <= 1'

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    (points, weights) = fcn(1)

    def exact_integral(p, q):
        return 0.5 / ((p + 1) * (q + 1) * (p + q + 2))

    def quadrature_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for p in range(2):
        for q in range(2 - p):
            exact = exact_integral(p, q)
            quad = quadrature_integral(p, q)
            assert np.isclose(quad, exact, rtol=1e-14, atol=1e-14), f'1-pt rule not exact for x^{p}y^{q}: got {quad}, expected {exact}'
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        exact = exact_integral(p, q)
        quad = quadrature_integral(p, q)
        assert not np.isclose(quad, exact, rtol=1e-10, atol=1e-14), f'1-pt rule should not be exact for x^{p}y^{q}'

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    (points, weights) = fcn(3)

    def exact_integral(p, q):
        return 0.5 / ((p + 1) * (q + 1) * (p + q + 2))

    def quadrature_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for p in range(3):
        for q in range(3 - p):
            exact = exact_integral(p, q)
            quad = quadrature_integral(p, q)
            assert np.isclose(quad, exact, rtol=1e-14, atol=1e-14), f'3-pt rule not exact for x^{p}y^{q}: got {quad}, expected {exact}'
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        exact = exact_integral(p, q)
        quad = quadrature_integral(p, q)
        assert not np.isclose(quad, exact, rtol=1e-10, atol=1e-14), f'3-pt rule should not be exact for x^{p}y^{q}'

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    (points, weights) = fcn(4)

    def exact_integral(p, q):
        return 0.5 / ((p + 1) * (q + 1) * (p + q + 2))

    def quadrature_integral(p, q):
        return np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
    for p in range(4):
        for q in range(4 - p):
            exact = exact_integral(p, q)
            quad = quadrature_integral(p, q)
            assert np.isclose(quad, exact, rtol=1e-14, atol=1e-14), f'4-pt rule not exact for x^{p}y^{q}: got {quad}, expected {exact}'
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        exact = exact_integral(p, q)
        quad = quadrature_integral(p, q)
        assert not np.isclose(quad, exact, rtol=1e-10, atol=1e-14), f'4-pt rule should not be exact for x^{p}y^{q}'