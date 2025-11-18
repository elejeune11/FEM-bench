def test_quad_quadrature_2D_invalid_inputs(fcn):
    with pytest.raises(ValueError):
        fcn(num_pts=2)
    with pytest.raises(ValueError):
        fcn(num_pts=5)
    with pytest.raises(ValueError):
        fcn(num_pts=10)

def test_quad_quadrature_2D_basics(fcn):
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == ((num_pts, 2) if num_pts > 1 else (1, 2))
        assert points.dtype == np.float64
        assert weights.shape == (num_pts,)
        assert weights.dtype == np.float64
        assert np.isclose(weights.sum(), 4.0)
        assert np.all(np.abs(points) <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    (points, weights) = fcn(num_pts=1)
    assert np.allclose(points, [[0.0, 0.0]])
    assert np.allclose(weights, [4.0])

    def integrate(poly):
        return poly(0, 0) * weights[0]
    a00 = 2.71828
    a10 = -3.14159
    a01 = 1.41421
    poly1 = lambda x, y: a00 + a10 * x + a01 * y
    analytic1 = 4 * a00
    assert np.isclose(integrate(poly1), analytic1)
    a20 = 1.0
    a02 = 1.0
    a11 = 1.0
    poly2 = lambda x, y: poly1(x, y) + a20 * x ** 2 + a02 * y ** 2 + a11 * x * y
    analytic2 = analytic1 + 4 * (a20 + a02)
    assert not np.isclose(integrate(poly2), analytic2)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    (points, weights) = fcn(num_pts=4)
    assert np.allclose(points, [[-1, -1], [1, -1], [-1, 1], [1, 1]])
    assert np.allclose(weights, [1, 1, 1, 1])

    def integrate(poly):
        return sum((poly(x, y) * w for (x, y, w) in zip(points[:, 0], points[:, 1], weights)))
    poly3 = lambda x, y: sum((a * x ** i * y ** j for i in range(4) for j in range(4) for a in [np.random.rand()]))
    analytic3 = integrate(poly3)
    assert np.isclose(integrate(poly3), analytic3)
    poly4 = lambda x, y: poly3(x, y) + x ** 4 + y ** 4
    analytic4 = integrate(poly3) + 8 / 5 + 8 / 5
    assert not np.isclose(integrate(poly4), analytic4)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    (points, weights) = fcn(num_pts=9)
    assert np.allclose(points, [[-np.sqrt(3 / 5), -np.sqrt(3 / 5)], [np.sqrt(3 / 5), -np.sqrt(3 / 5)], [-np.sqrt(3 / 5), np.sqrt(3 / 5)], [np.sqrt(3 / 5), np.sqrt(3 / 5)], [0, -np.sqrt(3 / 5)], [0, np.sqrt(3 / 5)], [-np.sqrt(3 / 5), 0], [np.sqrt(3 / 5), 0], [0, 0]])
    assert np.allclose(weights, [5 / 9, 5 / 9, 5 / 9, 5 / 9, 8 / 9, 8 / 9, 8 / 9, 8 / 9, 64 / 81])

    def integrate(poly):
        return sum((poly(x, y) * w for (x, y, w) in zip(points[:, 0], points[:, 1], weights)))
    poly5 = lambda x, y: sum((a * x ** i * y ** j for i in range(6) for j in range(6) for a in [np.random.rand()]))
    analytic5 = integrate(poly5)
    assert np.isclose(integrate(poly5), analytic5)
    poly6 = lambda x, y: poly5(x, y) + x ** 6 + y ** 6 + x ** 4 * y ** 2
    analytic6 = integrate(poly5) + 16 / 7 + 16 / 7 + 16 / 35
    assert not np.isclose(integrate(poly6), analytic6)