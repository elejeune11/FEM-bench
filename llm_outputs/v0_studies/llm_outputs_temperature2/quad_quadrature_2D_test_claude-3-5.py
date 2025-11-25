def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    import pytest
    for n in [-1, 0, 2, 3, 5, 8, 10]:
        with pytest.raises(ValueError):
            fcn(n)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    import numpy as np
    for n in [1, 4, 9]:
        (points, weights) = fcn(n)
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.abs(np.sum(weights) - 4.0) < 1e-14
        assert np.all(points >= -1.0)
        assert np.all(points <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    """
    import numpy as np
    (points, weights) = fcn(1)
    np.random.seed(42)
    a00 = np.random.randn()
    a10 = np.random.randn()
    a01 = np.random.randn()

    def f1(x, y):
        return a00 + a10 * x + a01 * y
    quad_result = np.sum(weights * f1(points[:, 0], points[:, 1]))
    exact_result = 4 * a00
    assert np.abs(quad_result - exact_result) < 1e-14

    def f2(x, y):
        return x * x + y * y + x * y
    quad_result = np.sum(weights * f2(points[:, 0], points[:, 1]))
    exact_result = 8 / 3
    assert np.abs(quad_result - exact_result) > 1e-10

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    import numpy as np
    (points, weights) = fcn(4)
    np.random.seed(42)
    coeffs = np.random.randn(4, 4)

    def f3(x, y):
        result = 0
        for i in range(4):
            for j in range(4):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * f3(points[:, 0], points[:, 1]))
    exact_result = 0
    for i in range(4):
        for j in range(4):
            if i % 2 == 0 and j % 2 == 0:
                exact_result += coeffs[i, j] * 4 * 2 ** i * 2 ** j / ((i + 1) * (j + 1))
    assert np.abs(quad_result - exact_result) < 1e-12

    def f4(x, y):
        return x ** 4 + y ** 4
    quad_result = np.sum(weights * f4(points[:, 0], points[:, 1]))
    exact_result = 32 / 5
    assert np.abs(quad_result - exact_result) > 1e-10

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    """
    import numpy as np
    (points, weights) = fcn(9)
    np.random.seed(42)
    coeffs = np.random.randn(6, 6)

    def f5(x, y):
        result = 0
        for i in range(6):
            for j in range(6):
                result += coeffs[i, j] * x ** i * y ** j
        return result
    quad_result = np.sum(weights * f5(points[:, 0], points[:, 1]))
    exact_result = 0
    for i in range(6):
        for j in range(6):
            if i % 2 == 0 and j % 2 == 0:
                exact_result += coeffs[i, j] * 4 * 2 ** i * 2 ** j / ((i + 1) * (j + 1))
    assert np.abs(quad_result - exact_result) < 1e-12

    def f6(x, y):
        return x ** 6 + y ** 6 + x ** 4 * y ** 2
    quad_result = np.sum(weights * f6(points[:, 0], points[:, 1]))
    exact_result = 32 / 7 + 32 / 7 + 32 / 45
    assert np.abs(quad_result - exact_result) > 1e-10