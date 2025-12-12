def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points."""
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(-1)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads."""
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(points >= -1) and np.all(points <= 1)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule."""
    (points, weights) = fcn(1)
    assert np.isclose(np.sum(weights * (points[:, 0] ** 0 * points[:, 1] ** 0)), 4.0)
    assert np.isclose(np.sum(weights * (points[:, 0] ** 1 * points[:, 1] ** 0)), 0.0)
    assert np.isclose(np.sum(weights * (points[:, 0] ** 0 * points[:, 1] ** 1)), 0.0)
    assert not np.isclose(np.sum(weights * (points[:, 0] ** 2 * points[:, 1] ** 0)), 4.0 / 3.0)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule."""
    (points, weights) = fcn(4)
    for i in range(4):
        for j in range(4):
            if i <= 3 and j <= 3:
                assert np.isclose(np.sum(weights * (points[:, 0] ** i * points[:, 1] ** j)), (1 + (-1) ** (i + 1)) / (i + 1) * (1 + (-1) ** (j + 1)) / (j + 1))
            else:
                assert not np.isclose(np.sum(weights * (points[:, 0] ** i * points[:, 1] ** j)), (1 + (-1) ** (i + 1)) / (i + 1) * (1 + (-1) ** (j + 1)) / (j + 1))

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule."""
    (points, weights) = fcn(9)
    for i in range(7):
        for j in range(7):
            if i <= 5 and j <= 5:
                assert np.isclose(np.sum(weights * (points[:, 0] ** i * points[:, 1] ** j)), (1 + (-1) ** (i + 1)) / (i + 1) * (1 + (-1) ** (j + 1)) / (j + 1))
            else:
                assert not np.isclose(np.sum(weights * (points[:, 0] ** i * points[:, 1] ** j)), (1 + (-1) ** (i + 1)) / (i + 1) * (1 + (-1) ** (j + 1)) / (j + 1))