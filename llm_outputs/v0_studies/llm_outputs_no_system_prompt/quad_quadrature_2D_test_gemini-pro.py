def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError."""
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(10)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1."""
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(weights.sum(), 4.0)
        assert np.all((-1 <= points) & (points <= 1))

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1]."""
    (points, weights) = fcn(1)
    rng = np.random.default_rng(0)
    (a00, a10, a01) = rng.random(3)
    analytic_integral = 4 * a00
    quadrature_approx = (a00 + a10 * points[:, 0] + a01 * points[:, 1]) @ weights
    assert np.isclose(quadrature_approx, analytic_integral)
    (a20, a02, a11) = rng.random(3)
    analytic_integral += 4 / 3 * (a20 + a02)
    quadrature_approx = (a00 + a10 * points[:, 0] + a01 * points[:, 1] + a20 * points[:, 0] ** 2 + a02 * points[:, 1] ** 2 + a11 * points[:, 0] * points[:, 1]) @ weights
    assert not np.isclose(quadrature_approx, analytic_integral)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1]."""
    (points, weights) = fcn(4)
    rng = np.random.default_rng(1)
    coeffs = rng.random((4, 4))
    poly = sum((coeffs[i, j] * points[:, 0] ** i * points[:, 1] ** j for i in range(4) for j in range(4)))
    analytic_integral = sum((coeffs[i, j] * 4 / (i + 1) / (j + 1) * (1 - (-1) ** (i + 1)) * (1 - (-1) ** (j + 1)) for i in range(4) for j in range(4)))
    quadrature_approx = poly @ weights
    assert np.isclose(quadrature_approx, analytic_integral)
    c40 = rng.random()
    analytic_integral += c40 * (4 / 5) * 2
    quadrature_approx = (poly + c40 * points[:, 0] ** 4) @ weights
    assert not np.isclose(quadrature_approx, analytic_integral)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1]."""
    (points, weights) = fcn(9)
    rng = np.random.default_rng(2)
    coeffs = rng.random((6, 6))
    poly = sum((coeffs[i, j] * points[:, 0] ** i * points[:, 1] ** j for i in range(6) for j in range(6)))
    analytic_integral = sum((coeffs[i, j] * 4 / (i + 1) / (j + 1) * (1 - (-1) ** (i + 1)) * (1 - (-1) ** (j + 1)) for i in range(6) for j in range(6)))
    quadrature_approx = poly @ weights
    assert np.isclose(quadrature_approx, analytic_integral)
    c60 = rng.random()
    analytic_integral += c60 * (4 / 7) * 2
    quadrature_approx = (poly + c60 * points[:, 0] ** 6) @ weights
    assert not np.isclose(quadrature_approx, analytic_integral)