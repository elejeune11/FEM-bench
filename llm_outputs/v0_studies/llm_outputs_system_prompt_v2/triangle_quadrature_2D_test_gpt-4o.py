def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(-1)

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
        assert np.all(points >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(1)
    (xi, eta) = points[0]
    assert np.isclose(np.sum(weights), 0.5)
    assert np.isclose(weights[0] * 1, 0.5)
    assert np.isclose(weights[0] * xi, 1 / 6)
    assert np.isclose(weights[0] * eta, 1 / 6)
    assert not np.isclose(weights[0] * xi ** 2, 1 / 12)
    assert not np.isclose(weights[0] * xi * eta, 1 / 24)
    assert not np.isclose(weights[0] * eta ** 2, 1 / 12)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(3)
    assert np.isclose(np.sum(weights), 0.5)
    for (xi, eta) in points:
        assert np.isclose(np.dot(weights, np.ones(3)), 0.5)
        assert np.isclose(np.dot(weights, xi), 1 / 6)
        assert np.isclose(np.dot(weights, eta), 1 / 6)
        assert np.isclose(np.dot(weights, xi ** 2), 1 / 12)
        assert np.isclose(np.dot(weights, xi * eta), 1 / 24)
        assert np.isclose(np.dot(weights, eta ** 2), 1 / 12)
        assert not np.isclose(np.dot(weights, xi ** 3), 1 / 20)
        assert not np.isclose(np.dot(weights, xi ** 2 * eta), 1 / 60)
        assert not np.isclose(np.dot(weights, xi * eta ** 2), 1 / 60)
        assert not np.isclose(np.dot(weights, eta ** 3), 1 / 20)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(4)
    assert np.isclose(np.sum(weights), 0.5)
    for (xi, eta) in points:
        assert np.isclose(np.dot(weights, np.ones(4)), 0.5)
        assert np.isclose(np.dot(weights, xi), 1 / 6)
        assert np.isclose(np.dot(weights, eta), 1 / 6)
        assert np.isclose(np.dot(weights, xi ** 2), 1 / 12)
        assert np.isclose(np.dot(weights, xi * eta), 1 / 24)
        assert np.isclose(np.dot(weights, eta ** 2), 1 / 12)
        assert np.isclose(np.dot(weights, xi ** 3), 1 / 20)
        assert np.isclose(np.dot(weights, xi ** 2 * eta), 1 / 60)
        assert np.isclose(np.dot(weights, xi * eta ** 2), 1 / 60)
        assert np.isclose(np.dot(weights, eta ** 3), 1 / 20)
        assert not np.isclose(np.dot(weights, xi ** 4), 1 / 30)
        assert not np.isclose(np.dot(weights, xi ** 3 * eta), 1 / 120)
        assert not np.isclose(np.dot(weights, xi ** 2 * eta ** 2), 1 / 180)
        assert not np.isclose(np.dot(weights, xi * eta ** 3), 1 / 120)
        assert not np.isclose(np.dot(weights, eta ** 4), 1 / 30)