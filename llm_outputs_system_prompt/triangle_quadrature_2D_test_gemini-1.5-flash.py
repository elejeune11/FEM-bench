def test_triangle_quadrature_2D_invalid_inputs(fcn):
    with raises(ValueError):
        fcn(num_pts=2)
    with raises(ValueError):
        fcn(num_pts=5)
    with raises(ValueError):
        fcn(num_pts=0)
    with raises(ValueError):
        fcn(num_pts=-1)

def test_triangle_quadrature_2D_basics(fcn):
    for num_pts in [1, 3, 4]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert points.dtype == np.float64
        assert weights.shape == (num_pts,)
        assert weights.dtype == np.float64
        assert np.isclose(weights.sum(), 0.5)
        assert np.all(points >= 0)
        assert np.all(points.sum(axis=1) <= 1)

def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    (points, weights) = fcn(num_pts=1)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    for (p, q) in monomials[:3]:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(approx, exact_integral(p, q))
    for (p, q) in monomials[3:]:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert not np.isclose(approx, exact_integral(p, q))

def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    (points, weights) = fcn(num_pts=3)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    monomials = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]
    for (p, q) in monomials[:6]:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(approx, exact_integral(p, q))
    for (p, q) in monomials[6:]:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert not np.isclose(approx, exact_integral(p, q))

def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    (points, weights) = fcn(num_pts=4)

    def exact_integral(p, q):
        return np.math.factorial(p) * np.math.factorial(q) / np.math.factorial(p + q + 2)
    monomials = [(p, q) for p in range(4) for q in range(4) if p + q <= 3]
    quartics = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    for (p, q) in monomials:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert np.isclose(approx, exact_integral(p, q))
    for (p, q) in quartics:
        approx = np.sum(weights * points[:, 0] ** p * points[:, 1] ** q)
        assert not np.isclose(approx, exact_integral(p, q))