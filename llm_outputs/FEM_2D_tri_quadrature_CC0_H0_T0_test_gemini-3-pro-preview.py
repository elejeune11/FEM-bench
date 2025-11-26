def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """
    Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 5, -1, 10]
    for n in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    for n in [1, 3, 4]:
        (points, weights) = fcn(n)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 0.5, atol=1e-15)
        x = points[:, 0]
        y = points[:, 1]
        assert np.all(x >= -1e-15)
        assert np.all(y >= -1e-15)
        assert np.all(x + y <= 1.0 + 1e-15)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Accuracy of the 1-point centroid rule.
    This rule is exact for total degree <= 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    n = 1
    monomials_exact = [(0, 0), (1, 0), (0, 1)]
    for (p, q) in monomials_exact:
        exact = _exact_integral(p, q)
        approx = _numerical_integral(fcn, n, p, q)
        assert np.isclose(approx, exact, atol=1e-15)
    monomials_fail = [(2, 0), (1, 1), (0, 2)]
    for (p, q) in monomials_fail:
        exact = _exact_integral(p, q)
        approx = _numerical_integral(fcn, n, p, q)
        assert not np.isclose(approx, exact, atol=1e-15)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """
    Accuracy of the classic 3-point rule.
    This rule is exact for total degree <= 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    n = 3
    monomials_exact = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for (p, q) in monomials_exact:
        exact = _exact_integral(p, q)
        approx = _numerical_integral(fcn, n, p, q)
        assert np.isclose(approx, exact, atol=1e-15)
    monomials_fail = [(3, 0), (2, 1), (1, 2), (0, 3)]
    for (p, q) in monomials_fail:
        exact = _exact_integral(p, q)
        approx = _numerical_integral(fcn, n, p, q)
        assert not np.isclose(approx, exact, atol=1e-15)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """
    Accuracy of the 4-point rule.
    This rule is exact for total degree <= 3. We verify exactness on all monomials with p+q <= 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    n = 4
    for degree in range(4):
        for p in range(degree + 1):
            q = degree - p
            exact = _exact_integral(p, q)
            approx = _numerical_integral(fcn, n, p, q)
            assert np.isclose(approx, exact, atol=1e-14)
    monomials_fail = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for (p, q) in monomials_fail:
        exact = _exact_integral(p, q)
        approx = _numerical_integral(fcn, n, p, q)
        assert not np.isclose(approx, exact, atol=1e-15)