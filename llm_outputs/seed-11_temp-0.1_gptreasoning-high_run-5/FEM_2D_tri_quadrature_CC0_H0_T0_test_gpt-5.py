def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """
    Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    for bad_n in [-10, -1, 0, 2, 5, 6, 7, 8, 9, 10]:
        with pytest.raises(ValueError):
            fcn(bad_n)

def test_triangle_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    for n in (1, 3, 4):
        pts, wts = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isfinite(pts).all()
        assert np.isfinite(wts).all()
        assert np.isclose(np.sum(wts), 0.5, rtol=0, atol=1e-15)
        x = pts[:, 0]
        y = pts[:, 1]
        tol = 1e-15
        assert np.all(x >= -tol)
        assert np.all(y >= -tol)
        assert np.all(x + y <= 1 + tol)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    pts, wts = fcn(1)
    tol = 1e-14
    for p, q in [(0, 0), (1, 0), (0, 1)]:
        approx = _quad_integral(pts, wts, p, q)
        exact = _exact_integral_monomial(p, q)
        assert np.isclose(approx, exact, rtol=0, atol=tol)
    for p, q in [(2, 0), (1, 1), (0, 2)]:
        approx = _quad_integral(pts, wts, p, q)
        exact = _exact_integral_monomial(p, q)
        assert abs(approx - exact) > 1e-08

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """
    Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    pts, wts = fcn(3)
    tol = 1e-12
    for p, q in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        approx = _quad_integral(pts, wts, p, q)
        exact = _exact_integral_monomial(p, q)
        assert np.isclose(approx, exact, rtol=0, atol=tol)
    for p, q in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        approx = _quad_integral(pts, wts, p, q)
        exact = _exact_integral_monomial(p, q)
        assert abs(approx - exact) > 1e-09

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """
    Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    pts, wts = fcn(4)
    tol = 1e-12
    for total in range(0, 4):
        for p in range(total + 1):
            q = total - p
            approx = _quad_integral(pts, wts, p, q)
            exact = _exact_integral_monomial(p, q)
            assert np.isclose(approx, exact, rtol=0, atol=tol)
    for p, q in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        approx = _quad_integral(pts, wts, p, q)
        exact = _exact_integral_monomial(p, q)
        assert abs(approx - exact) > 1e-09