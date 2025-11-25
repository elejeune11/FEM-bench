def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    for n in [-5, -1, 0, 2, 5, 6, 7, 10]:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    for n in (1, 3, 4):
        (pts, wts) = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isclose(np.sum(wts), 0.5, atol=1e-14)
        x = pts[:, 0]
        y = pts[:, 1]
        assert np.all(x >= -1e-15)
        assert np.all(y >= -1e-15)
        assert np.all(x + y <= 1 + 1e-15)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (pts, wts) = fcn(1)

    def quad(p, q):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.sum(wts * x ** p * y ** q))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        val = quad(p, q)
        ex = exact(p, q)
        assert np.isclose(val, ex, atol=1e-14, rtol=0)
    errors = []
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        val = quad(p, q)
        ex = exact(p, q)
        errors.append(abs(val - ex))
    assert any((err > 1e-06 for err in errors))

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (pts, wts) = fcn(3)

    def quad(p, q):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.sum(wts * x ** p * y ** q))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    for (p, q) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        val = quad(p, q)
        ex = exact(p, q)
        assert np.isclose(val, ex, atol=1e-14, rtol=0)
    errors = []
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        val = quad(p, q)
        ex = exact(p, q)
        errors.append(abs(val - ex))
    assert any((err > 1e-08 for err in errors))

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (pts, wts) = fcn(4)

    def quad(p, q):
        x = pts[:, 0]
        y = pts[:, 1]
        return float(np.sum(wts * x ** p * y ** q))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    exact_pairs = []
    for p in range(0, 4):
        for q in range(0, 4 - p):
            exact_pairs.append((p, q))
    for (p, q) in exact_pairs:
        val = quad(p, q)
        ex = exact(p, q)
        assert np.isclose(val, ex, atol=1e-12, rtol=0)
    errors = []
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        val = quad(p, q)
        ex = exact(p, q)
        errors.append(abs(val - ex))
    assert any((err > 1e-10 for err in errors))