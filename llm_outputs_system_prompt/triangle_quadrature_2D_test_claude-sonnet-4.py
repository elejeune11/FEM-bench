def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 5, -1, 10, 100]
    for num_pts in invalid_inputs:
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
        (x, y) = (points[:, 0], points[:, 1])
        assert np.all(x >= 0)
        assert np.all(y >= 0)
        assert np.all(x + y <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(1)
    (x, y) = (points[:, 0], points[:, 1])
    quad_result = np.sum(weights * 1)
    exact_result = 0.5
    assert np.isclose(quad_result, exact_result)
    quad_result = np.sum(weights * x)
    exact_result = factorial(1) * factorial(0) / factorial(3)
    assert np.isclose(quad_result, exact_result)
    quad_result = np.sum(weights * y)
    exact_result = factorial(0) * factorial(1) / factorial(3)
    assert np.isclose(quad_result, exact_result)
    quad_result = np.sum(weights * x ** 2)
    exact_result = factorial(2) * factorial(0) / factorial(4)
    assert not np.isclose(quad_result, exact_result)
    quad_result = np.sum(weights * x * y)
    exact_result = factorial(1) * factorial(1) / factorial(4)
    assert not np.isclose(quad_result, exact_result)
    quad_result = np.sum(weights * y ** 2)
    exact_result = factorial(0) * factorial(2) / factorial(4)
    assert not np.isclose(quad_result, exact_result)

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(3)
    (x, y) = (points[:, 0], points[:, 1])
    monomials_exact = [(1, 0.5), (x, factorial(1) * factorial(0) / factorial(3)), (y, factorial(0) * factorial(1) / factorial(3)), (x ** 2, factorial(2) * factorial(0) / factorial(4)), (x * y, factorial(1) * factorial(1) / factorial(4)), (y ** 2, factorial(0) * factorial(2) / factorial(4))]
    for (monomial, exact_result) in monomials_exact:
        quad_result = np.sum(weights * monomial)
        assert np.isclose(quad_result, exact_result)
    monomials_not_exact = [(x ** 3, factorial(3) * factorial(0) / factorial(5)), (x ** 2 * y, factorial(2) * factorial(1) / factorial(5)), (x * y ** 2, factorial(1) * factorial(2) / factorial(5)), (y ** 3, factorial(0) * factorial(3) / factorial(5))]
    for (monomial, exact_result) in monomials_not_exact:
        quad_result = np.sum(weights * monomial)
        assert not np.isclose(quad_result, exact_result)

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    (points, weights) = fcn(4)
    (x, y) = (points[:, 0], points[:, 1])
    monomials_exact = [(1, 0.5), (x, factorial(1) * factorial(0) / factorial(3)), (y, factorial(0) * factorial(1) / factorial(3)), (x ** 2, factorial(2) * factorial(0) / factorial(4)), (x * y, factorial(1) * factorial(1) / factorial(4)), (y ** 2, factorial(0) * factorial(2) / factorial(4)), (x ** 3, factorial(3) * factorial(0) / factorial(5)), (x ** 2 * y, factorial(2) * factorial(1) / factorial(5)), (x * y ** 2, factorial(1) * factorial(2) / factorial(5)), (y ** 3, factorial(0) * factorial(3) / factorial(5))]
    for (monomial, exact_result) in monomials_exact:
        quad_result = np.sum(weights * monomial)
        assert np.isclose(quad_result, exact_result)
    monomials_not_exact = [(x ** 4, factorial(4) * factorial(0) / factorial(6)), (x ** 3 * y, factorial(3) * factorial(1) / factorial(6)), (x ** 2 * y ** 2, factorial(2) * factorial(2) / factorial(6)), (x * y ** 3, factorial(1) * factorial(3) / factorial(6)), (y ** 4, factorial(0) * factorial(4) / factorial(6))]
    for (monomial, exact_result) in monomials_not_exact:
        quad_result = np.sum(weights * monomial)
        assert not np.isclose(quad_result, exact_result)