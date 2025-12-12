def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    import pytest
    invalid_values = [0, 2, 5, -1, 10]
    for v in invalid_values:
        with pytest.raises(ValueError):
            fcn(v)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1.
    """
    import numpy as np
    tol = 1e-12
    for n in (1, 3, 4):
        (points, weights) = fcn(n)
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(weights.sum(), 0.5, atol=tol, rtol=0)
        x = points[:, 0]
        y = points[:, 1]
        assert np.all(x >= -tol)
        assert np.all(y >= -tol)
        assert np.all(x + y <= 1 + tol)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    import numpy as np
    import math
    tol = 1e-12
    (points, weights) = fcn(1)

    def approx(p, q):
        vals = points[:, 0] ** p * points[:, 1] ** q
        return float(np.dot(weights, vals))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    for (p, q) in [(0, 0), (1, 0), (0, 1)]:
        a = approx(p, q)
        e = exact(p, q)
        assert np.isclose(a, e, atol=tol, rtol=1e-12)
    for (p, q) in [(2, 0), (1, 1), (0, 2)]:
        a = approx(p, q)
        e = exact(p, q)
        assert not np.isclose(a, e, atol=tol, rtol=1e-12)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    import numpy as np
    import math
    tol = 1e-12
    (points, weights) = fcn(3)

    def approx(p, q):
        vals = points[:, 0] ** p * points[:, 1] ** q
        return float(np.dot(weights, vals))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    exact_exponents = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for (p, q) in exact_exponents:
        a = approx(p, q)
        e = exact(p, q)
        assert np.isclose(a, e, atol=tol, rtol=1e-12)
    nonexact_exponents = [(3, 0), (2, 1), (1, 2), (0, 3)]
    for (p, q) in nonexact_exponents:
        a = approx(p, q)
        e = exact(p, q)
        assert not np.isclose(a, e, atol=tol, rtol=1e-12)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    import numpy as np
    import math
    tol = 1e-12
    (points, weights) = fcn(4)

    def approx(p, q):
        vals = points[:, 0] ** p * points[:, 1] ** q
        return float(np.dot(weights, vals))

    def exact(p, q):
        return math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
    for total in range(0, 4):
        for p in range(0, total + 1):
            q = total - p
            a = approx(p, q)
            e = exact(p, q)
            assert np.isclose(a, e, atol=tol, rtol=1e-12)
    nonexact_exponents = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for (p, q) in nonexact_exponents:
        a = approx(p, q)
        e = exact(p, q)
        assert not np.isclose(a, e, atol=tol, rtol=1e-12)