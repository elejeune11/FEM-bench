def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError."""
    for invalid_num in [-1, 0, 2, 5, 10]:
        with pytest.raises(ValueError):
            fcn(invalid_num)

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
        assert np.allclose(np.sum(weights), 0.5)
        assert np.all(points >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(1)
    monomials_linear = [(lambda x, y: 1, 1 / 2), (lambda x, y: x, 1 / 6), (lambda x, y: y, 1 / 6)]
    for (func, exact) in monomials_linear:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert np.allclose(computed, exact)
    monomials_quadratic = [(lambda x, y: x ** 2, 1 / 12), (lambda x, y: x * y, 1 / 24), (lambda x, y: y ** 2, 1 / 12)]
    for (func, exact) in monomials_quadratic:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert not np.allclose(computed, exact)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(3)
    monomials_quadratic = [(lambda x, y: 1, 1 / 2), (lambda x, y: x, 1 / 6), (lambda x, y: y, 1 / 6), (lambda x, y: x ** 2, 1 / 12), (lambda x, y: x * y, 1 / 24), (lambda x, y: y ** 2, 1 / 12)]
    for (func, exact) in monomials_quadratic:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert np.allclose(computed, exact, atol=1e-14)
    monomials_cubic = [(lambda x, y: x ** 3, 1 / 20), (lambda x, y: x ** 2 * y, 1 / 60), (lambda x, y: x * y ** 2, 1 / 60), (lambda x, y: y ** 3, 1 / 20)]
    for (func, exact) in monomials_cubic:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert not np.allclose(computed, exact)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!."""
    (points, weights) = fcn(4)
    monomials_cubic = [(lambda x, y: 1, 1 / 2), (lambda x, y: x, 1 / 6), (lambda x, y: y, 1 / 6), (lambda x, y: x ** 2, 1 / 12), (lambda x, y: x * y, 1 / 24), (lambda x, y: y ** 2, 1 / 12), (lambda x, y: x ** 3, 1 / 20), (lambda x, y: x ** 2 * y, 1 / 60), (lambda x, y: x * y ** 2, 1 / 60), (lambda x, y: y ** 3, 1 / 20)]
    for (func, exact) in monomials_cubic:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert np.allclose(computed, exact, atol=1e-14)
    monomials_quartic = [(lambda x, y: x ** 4, 1 / 30), (lambda x, y: x ** 3 * y, 1 / 105), (lambda x, y: x ** 2 * y ** 2, 1 / 180), (lambda x, y: x * y ** 3, 1 / 105), (lambda x, y: y ** 4, 1 / 30)]
    for (func, exact) in monomials_quartic:
        computed = np.sum(weights * func(points[:, 0], points[:, 1]))
        assert not np.allclose(computed, exact)