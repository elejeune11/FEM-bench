def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """Test that triangle_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    invalid_inputs = [0, 2, 5, 6, -1, 10, 100]
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
        assert points.shape == (num_pts, 2), f'Points shape mismatch for {num_pts} points'
        assert weights.shape == (num_pts,), f'Weights shape mismatch for {num_pts} points'
        assert points.dtype == np.float64, f'Points dtype should be float64, got {points.dtype}'
        assert weights.dtype == np.float64, f'Weights dtype should be float64, got {weights.dtype}'
        assert np.isclose(np.sum(weights), 0.5), f'Weights sum to {np.sum(weights)}, expected 0.5'
        x = points[:, 0]
        y = points[:, 1]
        assert np.all(x >= -1e-14), f'Some x coordinates are negative: {x}'
        assert np.all(y >= -1e-14), f'Some y coordinates are negative: {y}'
        assert np.all(x + y <= 1.0 + 1e-14), f'Some points violate x + y <= 1: {x + y}'

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """Accuracy of the 1-point centroid rule.
    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    """
    (points, weights) = fcn(1)
    exact_integrals = {(0, 0): 0.5, (1, 0): 1.0 / 6.0, (0, 1): 1.0 / 6.0}
    for ((p, q), exact) in exact_integrals.items():
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(computed, exact, atol=1e-14), f'Monomial x^{p}y^{q}: computed {computed}, expected {exact}'
    non_exact_monomials = [(2, 0), (1, 1), (0, 2)]
    for (p, q) in non_exact_monomials:
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        exact = 1.0 / ((p + 1) * (q + 1) * (p + q + 3))
        assert not np.isclose(computed, exact, atol=1e-10), f'Monomial x^{p}y^{q} should not be exact for 1-point rule'

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """Accuracy of the classic 3-point rule.
    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    """
    (points, weights) = fcn(3)
    exact_integrals = {(0, 0): 0.5, (1, 0): 1.0 / 6.0, (0, 1): 1.0 / 6.0, (2, 0): 1.0 / 12.0, (1, 1): 1.0 / 24.0, (0, 2): 1.0 / 12.0}
    for ((p, q), exact) in exact_integrals.items():
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(computed, exact, atol=1e-14), f'Monomial x^{p}y^{q}: computed {computed}, expected {exact}'
    non_exact_monomials = [(3, 0), (2, 1), (1, 2), (0, 3)]
    for (p, q) in non_exact_monomials:
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        exact = 1.0 / ((p + 1) * (q + 1) * (p + q + 3))
        assert not np.isclose(computed, exact, atol=1e-10), f'Monomial x^{p}y^{q} should not be exact for 3-point rule'

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """Accuracy of the 4-point rule.
    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    """
    (points, weights) = fcn(4)
    exact_integrals = {(0, 0): 0.5, (1, 0): 1.0 / 6.0, (0, 1): 1.0 / 6.0, (2, 0): 1.0 / 12.0, (1, 1): 1.0 / 24.0, (0, 2): 1.0 / 12.0, (3, 0): 1.0 / 20.0, (2, 1): 1.0 / 60.0, (1, 2): 1.0 / 60.0, (0, 3): 1.0 / 20.0}
    for ((p, q), exact) in exact_integrals.items():
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(computed, exact, atol=1e-13), f'Monomial x^{p}y^{q}: computed {computed}, expected {exact}'
    non_exact_monomials = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for (p, q) in non_exact_monomials:
        computed = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        exact = 1.0 / ((p + 1) * (q + 1) * (p + q + 3))
        assert not np.isclose(computed, exact, atol=1e-10), f'Monomial x^{p}y^{q} should not be exact for 4-point rule'