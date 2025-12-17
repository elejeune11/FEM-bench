def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_values = [-5, -1, 0, 2, 5, 6, 7, 8, 9, 10]
    for n in invalid_values:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
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
        assert np.isclose(wts.sum(), 0.5, rtol=0, atol=1e-15)
        x = pts[:, 0]
        y = pts[:, 1]
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert np.all(x >= -1e-15)
        assert np.all(y >= -1e-15)
        assert np.all(x + y <= 1 + 1e-15)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    pts, wts = fcn(1)

    def quad(p, q):
        return float(np.dot(wts, pts[:, 0] ** p * pts[:, 1] ** q))

    def exact(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))
    for p, q in [(0, 0), (1, 0), (0, 1)]:
        assert np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)
    for p, q in [(2, 0), (1, 1), (0, 2)]:
        assert not np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    pts, wts = fcn(3)

    def quad(p, q):
        return float(np.dot(wts, pts[:, 0] ** p * pts[:, 1] ** q))

    def exact(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))
    for p, q in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        assert np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)
    for p, q in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        assert not np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    pts, wts = fcn(4)

    def quad(p, q):
        return float(np.dot(wts, pts[:, 0] ** p * pts[:, 1] ** q))

    def exact(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))
    degree3_monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    for p, q in degree3_monomials:
        assert np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)
    for p, q in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        assert not np.isclose(quad(p, q), exact(p, q), rtol=1e-14, atol=1e-14)