def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError."""
    invalid_values = [0, 2, 5, -1, 10, 100]
    for num_pts in invalid_values:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_triangle_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule.
    For each supported rule (1, 3, 4 points):
      x >= 0, y >= 0, and x + y <= 1."""
    supported_num_pts = [1, 3, 4]
    for num_pts in supported_num_pts:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts}-point rule'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts}-point rule'
        assert points.dtype == np.float64, f'Points dtype should be float64 for {num_pts}-point rule'
        assert weights.dtype == np.float64, f'Weights dtype should be float64 for {num_pts}-point rule'
        assert np.isclose(np.sum(weights), 0.5), f'Weights should sum to 0.5 for {num_pts}-point rule'
        x = points[:, 0]
        y = points[:, 1]
        assert np.all(x >= -1e-14), f'All x coordinates should be >= 0 for {num_pts}-point rule'
        assert np.all(y >= -1e-14), f'All y coordinates should be >= 0 for {num_pts}-point rule'
        assert np.all(x + y <= 1 + 1e-14), f'All points should satisfy x + y <= 1 for {num_pts}-point rule'

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}."""
    (points, weights) = fcn(1)
    (x, y) = (points[:, 0], points[:, 1])
    exact_integrals = {(0, 0): 0.5, (1, 0): 1 / 6, (0, 1): 1 / 6, (2, 0): 1 / 12, (1, 1): 1 / 24, (0, 2): 1 / 12}
    for ((p, q), exact) in exact_integrals.items():
        if p + q <= 1:
            numerical = np.sum(weights * x ** p * y ** q)
            assert np.isclose(numerical, exact, rtol=1e-12), f'1-point rule should be exact for x^{p} y^{q}'
    quadratic_monomials = [(2, 0), (1, 1), (0, 2)]
    errors = []
    for (p, q) in quadratic_monomials:
        numerical = np.sum(weights * x ** p * y ** q)
        exact = exact_integrals[p, q]
        errors.append(abs(numerical - exact))
    assert any((e > 1e-10 for e in errors)), '1-point rule should NOT be exact for all quadratic monomials'

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}."""
    (points, weights) = fcn(3)
    (x, y) = (points[:, 0], points[:, 1])
    exact_integrals = {(0, 0): 0.5, (1, 0): 1 / 6, (0, 1): 1 / 6, (2, 0): 1 / 12, (1, 1): 1 / 24, (0, 2): 1 / 12, (3, 0): 1 / 20, (2, 1): 1 / 60, (1, 2): 1 / 60, (0, 3): 1 / 20}
    for ((p, q), exact) in exact_integrals.items():
        if p + q <= 2:
            numerical = np.sum(weights * x ** p * y ** q)
            assert np.isclose(numerical, exact, rtol=1e-12), f'3-point rule should be exact for x^{p} y^{q}'
    cubic_monomials = [(3, 0), (2, 1), (1, 2), (0, 3)]
    errors = []
    for (p, q) in cubic_monomials:
        numerical = np.sum(weights * x ** p * y ** q)
        exact = exact_integrals[p, q]
        errors.append(abs(numerical - exact))
    assert any((e > 1e-10 for e in errors)), '3-point rule should NOT be exact for all cubic monomials'

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}."""
    (points, weights) = fcn(4)
    (x, y) = (points[:, 0], points[:, 1])

    def factorial(n):
        if n <= 1:
            return 1
        return n * factorial(n - 1)

    def exact_integral(p, q):
        return factorial(p) * factorial(q) / factorial(p + q + 2)
    for p in range(4):
        for q in range(4 - p):
            exact = exact_integral(p, q)
            numerical = np.sum(weights * x ** p * y ** q)
            assert np.isclose(numerical, exact, rtol=1e-12), f'4-point rule should be exact for x^{p} y^{q}'
    quartic_monomials = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    errors = []
    for (p, q) in quartic_monomials:
        numerical = np.sum(weights * x ** p * y ** q)
        exact = exact_integral(p, q)
        errors.append(abs(numerical - exact))
    assert any((e > 1e-10 for e in errors)), '4-point rule should NOT be exact for all quartic monomials'