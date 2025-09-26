def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(8)
    with pytest.raises(ValueError):
        fcn(10)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(points >= -1) and np.all(points <= 1)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    """
    (points, weights) = fcn(1)
    (xi, eta) = points[0]
    weight = weights[0]
    (a00, a10, a01) = np.random.rand(3)
    integral_exact = 4 * (a00 + a10 * xi + a01 * eta)
    integral_approx = weight * (a00 + a10 * xi + a01 * eta)
    assert np.isclose(integral_exact, integral_approx)
    (a20, a02, a11) = np.random.rand(3)
    integral_exact = 4 * (a00 + a10 * xi + a01 * eta + a20 * xi ** 2 + a02 * eta ** 2 + a11 * xi * eta)
    integral_approx = weight * (a00 + a10 * xi + a01 * eta + a20 * xi ** 2 + a02 * eta ** 2 + a11 * xi * eta)
    assert not np.isclose(integral_exact, integral_approx)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(4)
    a = np.random.rand(16).reshape(4, 4)
    integral_exact = 0
    for i in range(4):
        for j in range(4):
            if i + j <= 3:
                integral_exact += a[i, j] * (2 ** (i + 1) - 1) * (2 ** (j + 1) - 1) / ((i + 1) * (j + 1))
    integral_approx = sum((weight * sum((a[i, j] * xi ** i * eta ** j for i in range(4) for j in range(4))) for ((xi, eta), weight) in zip(points, weights)))
    assert np.isclose(integral_exact, integral_approx)
    (a40, a04) = np.random.rand(2)
    integral_exact += a40 * (2 ** 5 - 1) / 5 + a04 * (2 ** 5 - 1) / 5
    integral_approx += sum((weight * (a40 * xi ** 4 + a04 * eta ** 4) for ((xi, eta), weight) in zip(points, weights)))
    assert not np.isclose(integral_exact, integral_approx)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    (points, weights) = fcn(9)
    a = np.random.rand(36).reshape(6, 6)
    integral_exact = 0
    for i in range(6):
        for j in range(6):
            if i + j <= 5:
                integral_exact += a[i, j] * (2 ** (i + 1) - 1) * (2 ** (j + 1) - 1) / ((i + 1) * (j + 1))
    integral_approx = sum((weight * sum((a[i, j] * xi ** i * eta ** j for i in range(6) for j in range(6))) for ((xi, eta), weight) in zip(points, weights)))
    assert np.isclose(integral_exact, integral_approx)
    (a60, a06, a42) = np.random.rand(3)
    integral_exact += a60 * (2 ** 7 - 1) / 7 + a06 * (2 ** 7 - 1) / 7 + a42 * (2 ** 5 - 1) / 5 * (2 ** 3 - 1) / 3
    integral_approx += sum((weight * (a60 * xi ** 6 + a06 * eta ** 6 + a42 * xi ** 4 * eta ** 2) for ((xi, eta), weight) in zip(points, weights)))
    assert not np.isclose(integral_exact, integral_approx)