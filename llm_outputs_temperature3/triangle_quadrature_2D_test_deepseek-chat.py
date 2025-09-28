def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(-1)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    for num_pts in [1, 3, 4]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.allclose(np.sum(weights), 0.5)
        assert np.all(points >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(1)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        monomial_vals = points[:, 0] ** p * points[:, 1] ** q
        quadrature_result = np.sum(weights * monomial_vals)
        exact_result = exact_integral(p, q)
        assert np.allclose(quadrature_result, exact_result)
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        monomial_vals = points[:, 0] ** p * points[:, 1] ** q
        quadrature_result = np.sum(weights * monomial_vals)
        exact_result = exact_integral(p, q)
        assert not np.allclose(quadrature_result, exact_result)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(3)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    for (p, q) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        monomial_vals = points[:, 0] ** p * points[:, 1] ** q
        quadrature_result = np.sum(weights * monomial_vals)
        exact_result = exact_integral(p, q)
        assert np.allclose(quadrature_result, exact_result)
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        monomial_vals = points[:, 0] ** p * points[:, 1] ** q
        quadrature_result = np.sum(weights * monomial_vals)
        exact_result = exact_integral(p, q)
        assert not np.allclose(quadrature_result, exact_result)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(4)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    for total_degree in range(4):
        for p in range(total_degree + 1):
            q = total_degree - p
            monomial_vals = points[:, 0] ** p * points[:, 1] ** q
            quadrature_result = np.sum(weights * monomial_vals)
            exact_result = exact_integral(p, q)
            assert np.allclose(quadrature_result, exact_result)
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        monomial_vals = points[:, 0] ** p * points[:, 1] ** q
        quadrature_result = np.sum(weights * monomial_vals)
        exact_result = exact_integral(p, q)
        assert not np.allclose(quadrature_result, exact_result)