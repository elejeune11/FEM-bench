def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError."""
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1."""
    for num_pts in [1, 3, 4]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 0.5)
        assert np.all(points >= 0) and np.all(points <= 1)
        assert np.all(np.sum(points, axis=1) <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}."""
    (points, weights) = fcn(1)

    def monomial(x, y, p, q):
        return x ** p * y ** q
    exact_values = [1, 1 / 3, 1 / 3, 1 / 6, 1 / 6, 1 / 6]
    monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for (i, (p, q)) in enumerate(monomials):
        integral_approx = np.sum(weights * monomial(points[:, 0], points[:, 1], p, q))
        integral_exact = exact_values[i]
        if i < 3:
            assert np.isclose(integral_approx, integral_exact)
        else:
            assert not np.isclose(integral_approx, integral_exact)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}."""
    (points, weights) = fcn(3)

    def monomial(x, y, p, q):
        return x ** p * y ** q
    exact_values = [1 / 2, 1 / 6, 1 / 6, 1 / 12, 1 / 24, 1 / 12, 1 / 20, 1 / 30, 1 / 30, 1 / 20]
    monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    for (i, (p, q)) in enumerate(monomials):
        integral_approx = np.sum(weights * monomial(points[:, 0], points[:, 1], p, q))
        integral_exact = exact_values[i]
        if i < 6:
            assert np.isclose(integral_approx, integral_exact)
        else:
            assert not np.isclose(integral_approx, integral_exact)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}."""
    (points, weights) = fcn(4)

    def monomial(x, y, p, q):
        return x ** p * y ** q
    exact_values = [1 / 2, 1 / 6, 1 / 6, 1 / 12, 1 / 24, 1 / 12, 1 / 20, 1 / 30, 1 / 30, 1 / 20, 1 / 30, 1 / 60, 1 / 60, 1 / 60, 1 / 30]
    monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for (i, (p, q)) in enumerate(monomials):
        integral_approx = np.sum(weights * monomial(points[:, 0], points[:, 1], p, q))
        integral_exact = exact_values[i]
        if i < 10:
            assert np.isclose(integral_approx, integral_exact)
        else:
            assert not np.isclose(integral_approx, integral_exact)