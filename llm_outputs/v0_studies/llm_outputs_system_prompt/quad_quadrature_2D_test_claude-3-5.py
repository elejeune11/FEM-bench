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
        assert isinstance(points, np.ndarray)
        assert isinstance(weights, np.ndarray)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert points.shape == (n_pts, 2)
        assert weights.shape == (n_pts,)
        assert np.abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points[:, 0] >= -1.0)
        assert np.all(points[:, 0] <= 1.0)
        assert np.all(points[:, 1] >= -1.0)
        assert np.all(points[:, 1] <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(1)
    np.random.seed(42)
    a00 = np.random.rand()
    a10 = np.random.rand()
    a01 = np.random.rand()

    def poly_deg1(x, y):
        return a00 + a10 * x + a01 * y
    exact = 4 * a00 + 0 * a10 + 0 * a01
    quad = np.sum(weights * poly_deg1(points[:, 0], points[:, 1]))
    assert np.abs(quad - exact) < 1e-14

    def poly_deg2(x, y):
        return x * x + y * y + x * y
    exact = 8 / 3
    quad = np.sum(weights * poly_deg2(points[:, 0], points[:, 1]))
    assert np.abs(quad - exact) > 1e-10

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(4)
    np.random.seed(42)
    coeffs = np.random.rand(4, 4)

    def poly_deg3(x, y):
        result = 0
        for i in range(4):
            for j in range(4):
                result += coeffs[i, j] * x ** i * y ** j
        return result

    def exact_integral_deg3():
        result = 0
        for i in range(4):
            for j in range(4):
                if i % 2 == 0 and j % 2 == 0:
                    result += coeffs[i, j] * 4 * 2 ** i * 2 ** j / ((i + 1) * (j + 1))
        return result
    quad = np.sum(weights * poly_deg3(points[:, 0], points[:, 1]))
    exact = exact_integral_deg3()
    assert np.abs(quad - exact) < 1e-12

    def poly_deg4(x, y):
        return x ** 4 + y ** 4
    quad = np.sum(weights * poly_deg4(points[:, 0], points[:, 1]))
    exact = 32 / 5
    assert np.abs(quad - exact) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule."""
    import numpy as np
    (points, weights) = fcn(9)
    np.random.seed(42)
    coeffs = np.random.rand(6, 6)

    def poly_deg5(x, y):
        result = 0
        for i in range(6):
            for j in range(6):
                result += coeffs[i, j] * x ** i * y ** j
        return result

    def exact_integral_deg5():
        result = 0
        for i in range(6):
            for j in range(6):
                if i % 2 == 0 and j % 2 == 0:
                    result += coeffs[i, j] * 4 * 2 ** i * 2 ** j / ((i + 1) * (j + 1))
        return result
    quad = np.sum(weights * poly_deg5(points[:, 0], points[:, 1]))
    exact = exact_integral_deg5()
    assert np.abs(quad - exact) < 1e-10

    def poly_deg6(x, y):
        return x ** 6 + y ** 6 + x ** 4 * y ** 2
    quad = np.sum(weights * poly_deg6(points[:, 0], points[:, 1]))
    exact = 32 / 7 + 32 / 7 + 32 / 15
    assert np.abs(quad - exact) > 1e-10