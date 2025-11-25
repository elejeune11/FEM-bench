def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points. The quadrature rule only supports 1, 3, or 4 integration points. Any other request should raise a ValueError."""
    invalid_values = [-10, -1, 0, 2, 5, 6, 7, 8, 9, 10, 100]
    for n in invalid_values:
        try:
            fcn(n)
        except ValueError:
            pass
        else:
            assert False, f'Expected ValueError for num_pts={n}'

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule. For each supported rule (1, 3, 4 points): - The returned points and weights arrays have the correct shapes and dtypes. - The weights sum to 1/2, which is the exact area of the reference triangle. - All quadrature points lie inside the reference triangle, i.e. x >= 0, y >= 0, and x + y <= 1."""
    tol = 1e-15
    for n in (1, 3, 4):
        (pts, wts) = fcn(n)
        assert hasattr(pts, 'shape') and hasattr(wts, 'shape')
        assert pts.shape == (n, 2)
        assert wts.shape == (n,)
        assert str(pts.dtype) == 'float64'
        assert str(wts.dtype) == 'float64'
        s = float(wts.sum())
        assert abs(s - 0.5) <= 1e-15
        for i in range(n):
            x = float(pts[i, 0])
            y = float(pts[i, 1])
            assert x >= -tol
            assert y >= -tol
            assert x + y <= 1.0 + tol
            assert 1.0 - x - y >= -tol

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule. This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y}, then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}. Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""

    def fact(k):
        v = 1
        for i in range(2, k + 1):
            v *= i
        return v

    def exact(p, q):
        return fact(p) * fact(q) / fact(p + q + 2)

    def quad(p, q):
        (pts, wts) = fcn(1)
        total = 0.0
        for i in range(wts.shape[0]):
            x = float(pts[i, 0])
            y = float(pts[i, 1])
            total += float(wts[i]) * x ** p * y ** q
        return total
    tol = 1e-14
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        assert abs(quad(p, q) - exact(p, q)) <= tol
    nonzero_tol = 1e-12
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        err = abs(quad(p, q) - exact(p, q))
        assert err > nonzero_tol

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule. This rule is exact for total degree ≤ 2. We verify exactness on monomials {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative cubic monomials {x^3, x^2 y, x y^2, y^3}. Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""

    def fact(k):
        v = 1
        for i in range(2, k + 1):
            v *= i
        return v

    def exact(p, q):
        return fact(p) * fact(q) / fact(p + q + 2)

    def quad(p, q):
        (pts, wts) = fcn(3)
        total = 0.0
        for i in range(wts.shape[0]):
            x = float(pts[i, 0])
            y = float(pts[i, 1])
            total += float(wts[i]) * x ** p * y ** q
        return total
    tol = 1e-14
    for (p, q) in [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]:
        assert abs(quad(p, q) - exact(p, q)) <= tol
    nonzero_tol = 1e-12
    for (p, q) in [(3, 0), (2, 1), (1, 2), (0, 3)]:
        err = abs(quad(p, q) - exact(p, q))
        assert err > nonzero_tol

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule. This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3, then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}. Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""

    def fact(k):
        v = 1
        for i in range(2, k + 1):
            v *= i
        return v

    def exact(p, q):
        return fact(p) * fact(q) / fact(p + q + 2)

    def quad(p, q):
        (pts, wts) = fcn(4)
        total = 0.0
        for i in range(wts.shape[0]):
            x = float(pts[i, 0])
            y = float(pts[i, 1])
            total += float(wts[i]) * x ** p * y ** q
        return total
    tol = 1e-14
    for p in range(0, 4):
        for q in range(0, 4 - p):
            assert abs(quad(p, q) - exact(p, q)) <= tol
    nonzero_tol = 1e-12
    for (p, q) in [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]:
        err = abs(quad(p, q) - exact(p, q))
        assert err > nonzero_tol