def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [-3, -1, 0, 2, 5, 7, 10, 100]
    for n in invalid_inputs:
        with pytest.raises(ValueError):
            fcn(n)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    Also verify the canonical order and values as specified in the API.
    """
    tol = 1e-15
    for n in (1, 3, 4):
        pts, wts = fcn(n)
        assert isinstance(pts, np.ndarray)
        assert isinstance(wts, np.ndarray)
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert pts.dtype == np.float64
        assert wts.dtype == np.float64
        assert np.isclose(np.sum(wts), 0.5, rtol=0.0, atol=1e-15)
        x = pts[:, 0]
        y = pts[:, 1]
        assert np.all(x >= -tol)
        assert np.all(y >= -tol)
        assert np.all(x + y <= 1 + tol)
    pts1_exp = np.array([[1 / 3, 1 / 3]], dtype=np.float64)
    wts1_exp = np.array([0.5], dtype=np.float64)
    pts1, wts1 = fcn(1)
    assert np.allclose(pts1, pts1_exp, rtol=0.0, atol=1e-15)
    assert np.allclose(wts1, wts1_exp, rtol=0.0, atol=1e-15)
    pts3_exp = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=np.float64)
    wts3_exp = np.array([1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
    pts3, wts3 = fcn(3)
    assert np.allclose(pts3, pts3_exp, rtol=0.0, atol=1e-15)
    assert np.allclose(wts3, wts3_exp, rtol=0.0, atol=1e-15)
    pts4_exp = np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]], dtype=np.float64)
    wts4_exp = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96], dtype=np.float64)
    pts4, wts4 = fcn(4)
    assert np.allclose(pts4, pts4_exp, rtol=0.0, atol=1e-15)
    assert np.allclose(wts4, wts4_exp, rtol=0.0, atol=1e-15)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    pts, wts = fcn(1)

    def exact_integral(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))

    def quad(p, q):
        return float(np.sum(wts * pts[:, 0] ** p * pts[:, 1] ** q))
    for p, q in [(0, 0), (1, 0), (0, 1)]:
        approx = quad(p, q)
        exact = exact_integral(p, q)
        assert abs(approx - exact) <= 1e-14
    for p, q in [(2, 0), (1, 1), (0, 2)]:
        approx = quad(p, q)
        exact = exact_integral(p, q)
        assert abs(approx - exact) > 1e-06

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    pts, wts = fcn(3)

    def exact_integral(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))

    def quad(p, q):
        return float(np.sum(wts * pts[:, 0] ** p * pts[:, 1] ** q))
    for p, q in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        approx = quad(p, q)
        exact = exact_integral(p, q)
        assert abs(approx - exact) <= 1e-14
    for p, q in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        approx = quad(p, q)
        exact = exact_integral(p, q)
        assert abs(approx - exact) > 1e-08

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    pts, wts = fcn(4)

    def exact_integral(p, q):
        return 1.0 / ((p + 1) * (q + 1) * (p + q + 2))

    def quad(p, q):
        return float(np.sum(wts * pts[:, 0] ** p * pts[:, 1] ** q))
    for p in range(0, 4):
        for q in range(0, 4 - p):
            approx = quad(p, q)
            exact = exact_integral(p, q)
            assert abs(approx - exact) <= 1e-14
    quartics = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for p, q in quartics:
        approx = quad(p, q)
        exact = exact_integral(p, q)
        assert abs(approx - exact) > 1e-08