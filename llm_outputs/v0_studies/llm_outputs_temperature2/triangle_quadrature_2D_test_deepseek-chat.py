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
        assert np.abs(np.sum(weights) - 0.5) < 1e-14
        assert np.all(points >= 0)
        assert np.all(points[:, 0] + points[:, 1] <= 1 + 1e-14)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(1)
    exact_1 = 1 / 2
    exact_x = 1 / 6
    exact_y = 1 / 6
    computed_1 = np.sum(weights)
    computed_x = np.sum(weights * points[:, 0])
    computed_y = np.sum(weights * points[:, 1])
    assert np.abs(computed_1 - exact_1) < 1e-14
    assert np.abs(computed_x - exact_x) < 1e-14
    assert np.abs(computed_y - exact_y) < 1e-14
    exact_x2 = 1 / 12
    exact_xy = 1 / 24
    exact_y2 = 1 / 12
    computed_x2 = np.sum(weights * points[:, 0] ** 2)
    computed_xy = np.sum(weights * points[:, 0] * points[:, 1])
    computed_y2 = np.sum(weights * points[:, 1] ** 2)
    assert np.abs(computed_x2 - exact_x2) > 0.001
    assert np.abs(computed_xy - exact_xy) > 0.001
    assert np.abs(computed_y2 - exact_y2) > 0.001

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(3)
    exact_1 = 1 / 2
    exact_x = 1 / 6
    exact_y = 1 / 6
    exact_x2 = 1 / 12
    exact_xy = 1 / 24
    exact_y2 = 1 / 12
    computed_1 = np.sum(weights)
    computed_x = np.sum(weights * points[:, 0])
    computed_y = np.sum(weights * points[:, 1])
    computed_x2 = np.sum(weights * points[:, 0] ** 2)
    computed_xy = np.sum(weights * points[:, 0] * points[:, 1])
    computed_y2 = np.sum(weights * points[:, 1] ** 2)
    assert np.abs(computed_1 - exact_1) < 1e-14
    assert np.abs(computed_x - exact_x) < 1e-14
    assert np.abs(computed_y - exact_y) < 1e-14
    assert np.abs(computed_x2 - exact_x2) < 1e-14
    assert np.abs(computed_xy - exact_xy) < 1e-14
    assert np.abs(computed_y2 - exact_y2) < 1e-14
    exact_x3 = 1 / 20
    exact_x2y = 1 / 60
    exact_xy2 = 1 / 60
    exact_y3 = 1 / 20
    computed_x3 = np.sum(weights * points[:, 0] ** 3)
    computed_x2y = np.sum(weights * points[:, 0] ** 2 * points[:, 1])
    computed_xy2 = np.sum(weights * points[:, 0] * points[:, 1] ** 2)
    computed_y3 = np.sum(weights * points[:, 1] ** 3)
    assert np.abs(computed_x3 - exact_x3) > 0.0001
    assert np.abs(computed_x2y - exact_x2y) > 0.0001
    assert np.abs(computed_xy2 - exact_xy2) > 0.0001
    assert np.abs(computed_y3 - exact_y3) > 0.0001

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(4)
    exact_1 = 1 / 2
    exact_x = 1 / 6
    exact_y = 1 / 6
    exact_x2 = 1 / 12
    exact_xy = 1 / 24
    exact_y2 = 1 / 12
    exact_x3 = 1 / 20
    exact_x2y = 1 / 60
    exact_xy2 = 1 / 60
    exact_y3 = 1 / 20
    computed_1 = np.sum(weights)
    computed_x = np.sum(weights * points[:, 0])
    computed_y = np.sum(weights * points[:, 1])
    computed_x2 = np.sum(weights * points[:, 0] ** 2)
    computed_xy = np.sum(weights * points[:, 0] * points[:, 1])
    computed_y2 = np.sum(weights * points[:, 1] ** 2)
    computed_x3 = np.sum(weights * points[:, 0] ** 3)
    computed_x2y = np.sum(weights * points[:, 0] ** 2 * points[:, 1])
    computed_xy2 = np.sum(weights * points[:, 0] * points[:, 1] ** 2)
    computed_y3 = np.sum(weights * points[:, 1] ** 3)
    assert np.abs(computed_1 - exact_1) < 1e-14
    assert np.abs(computed_x - exact_x) < 1e-14
    assert np.abs(computed_y - exact_y) < 1e-14
    assert np.abs(computed_x2 - exact_x2) < 1e-14
    assert np.abs(computed_xy - exact_xy) < 1e-14
    assert np.abs(computed_y2 - exact_y2) < 1e-14
    assert np.abs(computed_x3 - exact_x3) < 1e-14
    assert np.abs(computed_x2y - exact_x2y) < 1e-14
    assert np.abs(computed_xy2 - exact_xy2) < 1e-14
    assert np.abs(computed_y3 - exact_y3) < 1e-14
    exact_x4 = 1 / 30
    exact_x3y = 1 / 120
    exact_x2y2 = 1 / 180
    exact_xy3 = 1 / 120
    exact_y4 = 1 / 30
    computed_x4 = np.sum(weights * points[:, 0] ** 4)
    computed_x3y = np.sum(weights * points[:, 0] ** 3 * points[:, 1])
    computed_x2y2 = np.sum(weights * points[:, 0] ** 2 * points[:, 1] ** 2)
    computed_xy3 = np.sum(weights * points[:, 0] * points[:, 1] ** 3)
    computed_y4 = np.sum(weights * points[:, 1] ** 4)
    assert np.abs(computed_x4 - exact_x4) > 1e-05
    assert np.abs(computed_x3y - exact_x3y) > 1e-05
    assert np.abs(computed_x2y2 - exact_x2y2) > 1e-05
    assert np.abs(computed_xy3 - exact_xy3) > 1e-05
    assert np.abs(computed_y4 - exact_y4) > 1e-05