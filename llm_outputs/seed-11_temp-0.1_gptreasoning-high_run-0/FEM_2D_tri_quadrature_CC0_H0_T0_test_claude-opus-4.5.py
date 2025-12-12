def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError."""
    invalid_inputs = [0, 2, 5, 6, 7, 10, -1, 100]
    for num_pts in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1."""
    for num_pts in [1, 3, 4]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts}-point rule'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts}-point rule'
        assert points.dtype == np.float64, f'Points dtype mismatch for {num_pts}-point rule'
        assert weights.dtype == np.float64, f'Weights dtype mismatch for {num_pts}-point rule'
        assert np.isclose(np.sum(weights), 0.5), f'Weights sum mismatch for {num_pts}-point rule'
        for i in range(num_pts):
            (x, y) = points[i]
            assert x >= -1e-14, f'Point {i} has x < 0 for {num_pts}-point rule'
            assert y >= -1e-14, f'Point {i} has y < 0 for {num_pts}-point rule'
            assert x + y <= 1 + 1e-14, f'Point {i} has x + y > 1 for {num_pts}-point rule'

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}."""
    (points, weights) = fcn(1)
    exact_monomials = [(0, 0), (1, 0), (0, 1)]
    for (p, q) in exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        assert np.isclose(approx, exact, rtol=1e-12), f'1-pt rule not exact for x^{p}*y^{q}'
    non_exact_monomials = [(2, 0), (1, 1), (0, 2)]
    non_exact_count = 0
    for (p, q) in non_exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        if not np.isclose(approx, exact, rtol=1e-10):
            non_exact_count += 1
    assert non_exact_count > 0, '1-pt rule should not be exact for all quadratic monomials'

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}."""
    (points, weights) = fcn(3)
    exact_monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for (p, q) in exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        assert np.isclose(approx, exact, rtol=1e-12), f'3-pt rule not exact for x^{p}*y^{q}'
    non_exact_monomials = [(3, 0), (2, 1), (1, 2), (0, 3)]
    non_exact_count = 0
    for (p, q) in non_exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        if not np.isclose(approx, exact, rtol=1e-10):
            non_exact_count += 1
    assert non_exact_count > 0, '3-pt rule should not be exact for all cubic monomials'

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}."""
    (points, weights) = fcn(4)
    exact_monomials = []
    for p in range(4):
        for q in range(4 - p):
            exact_monomials.append((p, q))
    for (p, q) in exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        assert np.isclose(approx, exact, rtol=1e-12), f'4-pt rule not exact for x^{p}*y^{q}'
    non_exact_monomials = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    non_exact_count = 0
    for (p, q) in non_exact_monomials:
        exact = _exact_integral_monomial(p, q)
        approx = _quadrature_integral(points, weights, p, q)
        if not np.isclose(approx, exact, rtol=1e-10):
            non_exact_count += 1
    assert non_exact_count > 0, '4-pt rule should not be exact for all quartic monomials'