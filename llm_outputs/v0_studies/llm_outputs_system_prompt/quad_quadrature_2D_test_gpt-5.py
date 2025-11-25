def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError."""
    invalid_values = [-3, -1, 0, 2, 3, 5, 6, 7, 8, 10, 11, 16]
    for n in invalid_values:
        try:
            fcn(n)
        except ValueError:
            continue
        assert False, f'Expected ValueError for num_pts={n}'

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1."""
    for n in (1, 4, 9):
        (points, weights) = fcn(n)
        assert hasattr(points, 'shape') and hasattr(weights, 'shape')
        assert points.shape == (n, 2)
        assert weights.shape == (n,)
        assert hasattr(points, 'dtype') and hasattr(weights, 'dtype')
        assert points.dtype.kind == 'f' and weights.dtype.kind == 'f'
        assert points.dtype.itemsize == 8 and weights.dtype.itemsize == 8
        total_weight = float(weights.sum())
        assert abs(total_weight - 4.0) < 1e-14
        xs = points[:, 0]
        ys = points[:, 1]
        assert bool((xs >= -1.0).all()) and bool((xs <= 1.0).all())
        assert bool((ys >= -1.0).all()) and bool((ys <= 1.0).all())

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].
    1) Tests exactness for an affine polynomial P(x,y) = a00 + a10·x + a01·y.
    2) Adds quadratic contributions (x², y², xy) and confirms non-exactness."""
    (points, weights) = fcn(1)
    x = float(points[0, 0])
    y = float(points[0, 1])
    w = float(weights[0])
    (a00, a10, a01) = (1.2, -2.3, 0.7)
    P1 = a00 + a10 * x + a01 * y
    quad_affine = w * P1
    exact_affine = 4.0 * a00
    assert abs(quad_affine - exact_affine) < 1e-14
    (b20, b02, b11) = (0.5, 0.75, -1.1)
    P2 = P1 + b20 * (x * x) + b02 * (y * y) + b11 * (x * y)
    quad_quadratic = w * P2
    exact_quadratic = exact_affine + 4.0 / 3.0 * (b20 + b02)
    assert abs(quad_quadratic - exact_quadratic) > 1e-12

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    1) Tests exactness for all monomials x^i y^j with 0 ≤ i,j ≤ 3.
    2) Adds quartic terms (x^4 and y^4) and confirms non-exactness."""
    (points, weights) = fcn(4)
    coeff = [[0.0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            coeff[i][j] = 1.0 + i * 0.3 + j * 0.7
    quad_val = 0.0
    for k in range(points.shape[0]):
        x = float(points[k, 0])
        y = float(points[k, 1])
        val = 0.0
        x_pow = [1.0, x, x * x, x * x * x]
        y_pow = [1.0, y, y * y, y * y * y]
        for i in range(4):
            for j in range(4):
                val += coeff[i][j] * x_pow[i] * y_pow[j]
        quad_val += float(weights[k]) * val

    def I_1D(p):
        if p % 2 == 1:
            return 0.0
        return 2.0 / (p + 1)
    exact_val = 0.0
    for i in range(4):
        for j in range(4):
            exact_val += coeff[i][j] * I_1D(i) * I_1D(j)
    assert abs(quad_val - exact_val) < 1e-13
    (c40, c04) = (1.234, -0.987)
    quad_val2 = quad_val
    exact_val2 = exact_val
    for k in range(points.shape[0]):
        x = float(points[k, 0])
        y = float(points[k, 1])
        val_add = c40 * x ** 4 + c04 * y ** 4
        quad_val2 += float(weights[k]) * val_add
    exact_val2 += c40 * I_1D(4) * I_1D(0) + c04 * I_1D(0) * I_1D(4)
    assert abs(quad_val2 - exact_val2) > 1e-12

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
    1) Tests exactness for all monomials x^i y^j with 0 ≤ i,j ≤ 5.
    2) Adds degree-6 contributions (x^6, y^6, x^4 y^2) and confirms non-exactness is detected."""
    (points, weights) = fcn(9)
    coeff = [[0.0 for _ in range(6)] for _ in range(6)]
    for i in range(6):
        for j in range(6):
            coeff[i][j] = 0.5 + 0.11 * i - 0.07 * j + 0.013 * (i + j)
    quad_val = 0.0
    for k in range(points.shape[0]):
        x = float(points[k, 0])
        y = float(points[k, 1])
        x_pow = [1.0, x, x * x, x * x * x, x ** 4, x ** 5]
        y_pow = [1.0, y, y * y, y * y * y, y ** 4, y ** 5]
        val = 0.0
        for i in range(6):
            for j in range(6):
                val += coeff[i][j] * x_pow[i] * y_pow[j]
        quad_val += float(weights[k]) * val

    def I_1D(p):
        if p % 2 == 1:
            return 0.0
        return 2.0 / (p + 1)
    exact_val = 0.0
    for i in range(6):
        for j in range(6):
            exact_val += coeff[i][j] * I_1D(i) * I_1D(j)
    assert abs(quad_val - exact_val) < 1e-13
    (c60, c06, c42) = (0.321, -0.654, 0.777)
    quad_val2 = quad_val
    exact_val2 = exact_val
    for k in range(points.shape[0]):
        x = float(points[k, 0])
        y = float(points[k, 1])
        val_add = c60 * x ** 6 + c06 * y ** 6 + c42 * x ** 4 * y ** 2
        quad_val2 += float(weights[k]) * val_add
    exact_val2 += c60 * I_1D(6) * I_1D(0) + c06 * I_1D(0) * I_1D(6) + c42 * I_1D(4) * I_1D(2)
    assert abs(quad_val2 - exact_val2) > 1e-12