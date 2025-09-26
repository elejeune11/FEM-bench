def test_triangle_quadrature_2D_invalid_inputs(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 5, -1, 10]
    for num_pts in invalid_inputs:
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
        assert np.isclose(np.sum(weights), 0.5)
        tol = 1e-15
        assert np.all(points >= -tol)
        assert np.all(np.sum(points, axis=1) <= 1.0 + tol)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(1)

    def exact_integral(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)

    def numerical_integral(p, q):
        (x, y) = (points[:, 0], points[:, 1])
        integrand = x ** p * y ** q
        return np.dot(weights, integrand)
    for total_degree in range(2):
        for p in range(total_degree + 1):
            q = total_degree - p
            assert np.isclose(numerical_integral(p, q), exact_integral(p, q))
    total_degree = 2
    for p in range(total_degree + 1):
        q = total_degree - p
        assert not np.isclose(numerical_integral(p, q), exact_integral(p, q))

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(3)

    def exact_integral(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)

    def numerical_integral(p, q):
        (x, y) = (points[:, 0], points[:, 1])
        integrand = x ** p * y ** q
        return np.dot(weights, integrand)
    for total_degree in range(3):
        for p in range(total_degree + 1):
            q = total_degree - p
            assert np.isclose(numerical_integral(p, q), exact_integral(p, q))
    total_degree = 3
    num_inexact = 0
    for p in range(total_degree + 1):
        q = total_degree - p
        if not np.isclose(numerical_integral(p, q), exact_integral(p, q)):
            num_inexact += 1
    assert num_inexact > 0

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn: Callable[[int], Tuple[np.ndarray, np.ndarray]]):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(4)

    def exact_integral(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)

    def numerical_integral(p, q):
        (x, y) = (points[:, 0], points[:, 1])
        integrand = x ** p * y ** q
        return np.dot(weights, integrand)
    for total_degree in range(4):
        for p in range(total_degree + 1):
            q = total_degree - p
            assert np.isclose(numerical_integral(p, q), exact_integral(p, q))
    total_degree = 4
    num_inexact = 0
    for p in range(total_degree + 1):
        q = total_degree - p
        if not np.isclose(numerical_integral(p, q), exact_integral(p, q)):
            num_inexact += 1
    assert num_inexact > 0