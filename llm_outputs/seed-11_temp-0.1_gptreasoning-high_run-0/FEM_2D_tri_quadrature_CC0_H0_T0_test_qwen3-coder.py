def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    for num_pts in [-1, 0, 2, 5, 10]:
        with pytest.raises(ValueError):
            fcn(num_pts)

def test_triangle_quadrature_2D_basics(fcn):
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
        for (xi, eta) in points:
            assert xi >= 0.0
            assert eta >= 0.0
            assert xi + eta <= 1.0

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    (points, weights) = fcn(1)
    integral_1 = np.sum(weights * (points[:, 0] ** 0 * points[:, 1] ** 0))
    integral_x = np.sum(weights * (points[:, 0] ** 1 * points[:, 1] ** 0))
    integral_y = np.sum(weights * (points[:, 0] ** 0 * points[:, 1] ** 1))
    assert np.isclose(integral_1, 0.5)
    assert np.isclose(integral_x, 1.0 / 6.0)
    assert np.isclose(integral_y, 1.0 / 6.0)
    integral_x2 = np.sum(weights * (points[:, 0] ** 2 * points[:, 1] ** 0))
    integral_xy = np.sum(weights * (points[:, 0] ** 1 * points[:, 1] ** 1))
    integral_y2 = np.sum(weights * (points[:, 0] ** 0 * points[:, 1] ** 2))
    exact_x2 = 1.0 / 12.0
    exact_xy = 1.0 / 24.0
    exact_y2 = 1.0 / 12.0
    assert not np.isclose(integral_x2, exact_x2)
    assert not np.isclose(integral_xy, exact_xy)
    assert not np.isclose(integral_y2, exact_y2)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    (points, weights) = fcn(3)
    monomials_exact = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    exact_values = {(0, 0): 0.5, (1, 0): 1.0 / 6.0, (0, 1): 1.0 / 6.0, (2, 0): 1.0 / 12.0, (1, 1): 1.0 / 24.0, (0, 2): 1.0 / 12.0}
    for (p, q) in monomials_exact:
        computed = np.sum(weights * (points[:, 0] ** p * points[:, 1] ** q))
        expected = exact_values[p, q]
        assert np.isclose(computed, expected), f'Failed for x^{p} y^{q}'
    monomials_non_exact = [(3, 0), (2, 1), (1, 2), (0, 3)]
    exact_non = {(3, 0): 1.0 / 20.0, (2, 1): 1.0 / 60.0, (1, 2): 1.0 / 60.0, (0, 3): 1.0 / 20.0}
    for (p, q) in monomials_non_exact:
        computed = np.sum(weights * (points[:, 0] ** p * points[:, 1] ** q))
        expected = exact_non[p, q]
        assert not np.isclose(computed, expected), f'Unexpected exactness for x^{p} y^{q}'

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    (points, weights) = fcn(4)
    monomials_exact = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    exact_values = {(0, 0): 0.5, (1, 0): 1.0 / 6.0, (0, 1): 1.0 / 6.0, (2, 0): 1.0 / 12.0, (1, 1): 1.0 / 24.0, (0, 2): 1.0 / 12.0, (3, 0): 1.0 / 20.0, (2, 1): 1.0 / 60.0, (1, 2): 1.0 / 60.0, (0, 3): 1.0 / 20.0}
    for (p, q) in monomials_exact:
        computed = np.sum(weights * (points[:, 0] ** p * points[:, 1] ** q))
        expected = exact_values[p, q]
        assert np.isclose(computed, expected), f'Failed for x^{p} y^{q}'
    monomials_non_exact = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    exact_non = {(4, 0): 1.0 / 30.0, (3, 1): 1.0 / 120.0, (2, 2): 1.0 / 360.0, (1, 3): 1.0 / 120.0, (0, 4): 1.0 / 30.0}
    for (p, q) in monomials_non_exact:
        computed = np.sum(weights * (points[:, 0] ** p * points[:, 1] ** q))
        expected = exact_non[p, q]
        assert not np.isclose(computed, expected), f'Unexpected exactness for x^{p} y^{q}'