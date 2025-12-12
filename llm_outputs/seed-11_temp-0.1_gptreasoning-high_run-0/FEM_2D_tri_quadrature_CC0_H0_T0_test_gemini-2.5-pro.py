def test_triangle_quadrature_2D_invalid_inputs(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_requests = [0, 2, 5, -1, 1.5, 'a']
    for num_pts in invalid_requests:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_triangle_quadrature_2D_basics(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
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
        assert np.sum(weights) == pytest.approx(0.5)
        (x, y) = (points[:, 0], points[:, 1])
        assert np.all(x >= 0.0)
        assert np.all(y >= 0.0)
        assert np.all(x + y <= 1.0)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    (points, weights) = fcn(1)
    (x, y) = (points[:, 0], points[:, 1])
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        integrand = x ** p * y ** q
        numerical_integral = np.sum(weights * integrand)
        exact_value = _exact_integral(p, q)
        assert numerical_integral == pytest.approx(exact_value)
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        integrand = x ** p * y ** q
        numerical_integral = np.sum(weights * integrand)
        exact_value = _exact_integral(p, q)
        assert numerical_integral != pytest.approx(exact_value)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    (points, weights) = fcn(3)
    (x, y) = (points[:, 0], points[:, 1])
    for (p, q) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        integrand = x ** p * y ** q
        numerical_integral = np.sum(weights * integrand)
        exact_value = _exact_integral(p, q)
        assert numerical_integral == pytest.approx(exact_value)
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        integrand = x ** p * y ** q
        numerical_integral = np.sum(weights * integrand)
        exact_value = _exact_integral(p, q)
        assert numerical_integral != pytest.approx(exact_value)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    (points, weights) = fcn(4)
    (x, y) = (points[:, 0], points[:, 1])
    for p in range(4):
        for q in range(4 - p):
            integrand = x ** p * y ** q
            numerical_integral = np.sum(weights * integrand)
            exact_value = _exact_integral(p, q)
            assert numerical_integral == pytest.approx(exact_value)
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        integrand = x ** p * y ** q
        numerical_integral = np.sum(weights * integrand)
        exact_value = _exact_integral(p, q)
        assert numerical_integral != pytest.approx(exact_value)