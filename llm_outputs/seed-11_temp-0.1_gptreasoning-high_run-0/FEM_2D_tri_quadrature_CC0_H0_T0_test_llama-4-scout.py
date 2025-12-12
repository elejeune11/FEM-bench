def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)

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
        assert np.all(points[:, 0] >= 0)
        assert np.all(points[:, 1] >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    (points, weights) = fcn(1)
    assert np.isclose(np.sum(weights), 0.5)
    assert np.isclose(np.sum(weights * points[:, 0]), 1 / 6)
    assert np.isclose(np.sum(weights * points[:, 1]), 1 / 6)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 2), 1 / 9)
    assert not np.isclose(np.sum(weights * points[:, 0] * points[:, 1]), 1 / 36)
    assert not np.isclose(np.sum(weights * points[:, 1] ** 2), 1 / 9)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    (points, weights) = fcn(3)
    assert np.isclose(np.sum(weights), 0.5)
    assert np.isclose(np.sum(weights * points[:, 0]), 1 / 6)
    assert np.isclose(np.sum(weights * points[:, 1]), 1 / 6)
    assert np.isclose(np.sum(weights * points[:, 0] ** 2), 1 / 15)
    assert np.isclose(np.sum(weights * points[:, 0] * points[:, 1]), 1 / 45)
    assert np.isclose(np.sum(weights * points[:, 1] ** 2), 1 / 15)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 3), 1 / 64)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 2 * points[:, 1]), 1 / 135)
    assert not np.isclose(np.sum(weights * points[:, 0] * points[:, 1] ** 2), 1 / 135)
    assert not np.isclose(np.sum(weights * points[:, 1] ** 3), 1 / 64)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    (points, weights) = fcn(4)
    assert np.isclose(np.sum(weights), 0.5)
    assert np.isclose(np.sum(weights * points[:, 0]), 1 / 3)
    assert np.isclose(np.sum(weights * points[:, 1]), 1 / 3)
    assert np.isclose(np.sum(weights * points[:, 0] ** 2), 11 / 90)
    assert np.isclose(np.sum(weights * points[:, 0] * points[:, 1]), 1 / 45)
    assert np.isclose(np.sum(weights * points[:, 1] ** 2), 11 / 90)
    assert np.isclose(np.sum(weights * points[:, 0] ** 3), 1 / 15)
    assert np.isclose(np.sum(weights * points[:, 0] ** 2 * points[:, 1]), 1 / 90)
    assert np.isclose(np.sum(weights * points[:, 0] * points[:, 1] ** 2), 1 / 90)
    assert np.isclose(np.sum(weights * points[:, 1] ** 3), 1 / 15)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 4), 1 / 125)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 3 * points[:, 1]), 1 / 375)
    assert not np.isclose(np.sum(weights * points[:, 0] ** 2 * points[:, 1] ** 2), 1 / 675)
    assert not np.isclose(np.sum(weights * points[:, 0] * points[:, 1] ** 3), 1 / 375)
    assert not np.isclose(np.sum(weights * points[:, 1] ** 4), 1 / 125)