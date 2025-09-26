def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points."""
    import pytest
    import numpy as np
    invalid_sizes = [-1, 0, 2, 3, 5, 8, 10, 100]
    for n in invalid_sizes:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads."""
    import numpy as np
    for n_pts in [1, 4, 9]:
        (points, weights) = fcn(n_pts)
        assert points.shape == (n_pts, 2)
        assert weights.shape == (n_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points >= -1.0)
        assert np.all(points <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(1)
    np.random.seed(42)
    a00 = np.random.rand()
    a10 = np.random.rand()
    a01 = np.random.rand()

    def p1(x, y):
        return a00 + a10 * x + a01 * y
    quad_result = np.sum(weights * p1(points[:, 0], points[:, 1]))
    exact_result = 4 * a00
    assert np.abs(quad_result - exact_result) < 1e-14

    def p2(x, y):
        return p1(x, y) + x ** 2 + y ** 2 + x * y
    quad_result = np.sum(weights * p2(points[:, 0], points[:, 1]))
    exact_result = 4 * a00 + 4 / 3
    assert np.abs(quad_result - exact_result) > 1e-10

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(4)
    np.random.seed(42)
    coeff = np.random.rand(4, 4)

    def p3(x, y):
        result = 0
        for i in range(4):
            for j in range(4):
                result += coeff[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * p3(points[:, 0], points[:, 1]))
    exact_result = 0
    for i in range(4):
        for j in range(4):
            if i % 2 == 0 and j % 2 == 0:
                exact_result += coeff[i, j] * 4 * 2 / (i + 1) * 2 / (j + 1)
    assert np.abs(quad_result - exact_result) < 1e-13

    def p4(x, y):
        return p3(x, y) + x ** 4 + y ** 4
    quad_result = np.sum(weights * p4(points[:, 0], points[:, 1]))
    exact_result += 4 * 2 / 5
    assert np.abs(quad_result - exact_result) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(9)
    np.random.seed(42)
    coeff = np.random.rand(6, 6)

    def p5(x, y):
        result = 0
        for i in range(6):
            for j in range(6):
                result += coeff[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * p5(points[:, 0], points[:, 1]))
    exact_result = 0
    for i in range(6):
        for j in range(6):
            if i % 2 == 0 and j % 2 == 0:
                exact_result += coeff[i, j] * 4 * 2 / (i + 1) * 2 / (j + 1)
    assert np.abs(quad_result - exact_result) < 1e-12

    def p6(x, y):
        return p5(x, y) + x ** 6 + y ** 6 + x ** 4 * y ** 2
    quad_result = np.sum(weights * p6(points[:, 0], points[:, 1]))
    exact_result += 4 * (2 / 7 + 2 / 7 + 2 / 5 * 2 / 3)
    assert np.abs(quad_result - exact_result) > 1e-10