def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError."""
    for num_pts in [-1, 0, 2, 5, 10]:
        with pytest.raises(ValueError):
            fcn(num_pts)

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
        assert np.allclose(weights.sum(), 0.5)
        assert np.all(points >= 0)
        assert np.all(points.sum(axis=1) <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(1)
    for p in range(2):
        for q in range(2 - p):
            monomial = points[:, 0] ** p * points[:, 1] ** q
            integral = (weights * monomial).sum()
            exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
            assert np.allclose(integral, exact)
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        monomial = points[:, 0] ** p * points[:, 1] ** q
        integral = (weights * monomial).sum()
        exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
        assert not np.allclose(integral, exact)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(3)
    for p in range(3):
        for q in range(3 - p):
            monomial = points[:, 0] ** p * points[:, 1] ** q
            integral = (weights * monomial).sum()
            exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
            assert np.allclose(integral, exact)
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        monomial = points[:, 0] ** p * points[:, 1] ** q
        integral = (weights * monomial).sum()
        exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
        assert not np.allclose(integral, exact)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(4)
    for p in range(4):
        for q in range(4 - p):
            monomial = points[:, 0] ** p * points[:, 1] ** q
            integral = (weights * monomial).sum()
            exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
            assert np.allclose(integral, exact)
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        monomial = points[:, 0] ** p * points[:, 1] ** q
        integral = (weights * monomial).sum()
        exact = np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
        assert not np.allclose(integral, exact)