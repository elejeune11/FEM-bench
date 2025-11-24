def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('Input must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('1D input must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('2D input must have shape (n, 2)')
    else:
        raise ValueError('Input must be 1D or 2D array')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input contains non-finite values')
    n = xi.shape[0]
    (x, y) = (xi[:, 0], xi[:, 1])
    xm = 1 - x
    xp = 1 + x
    ym = 1 - y
    yp = 1 + y
    x2 = x * x
    y2 = y * y
    N = np.zeros((n, 8, 1))
    N[:, 0, 0] = -0.25 * xm * ym * (x + y + 1)
    N[:, 1, 0] = 0.25 * xp * ym * (x - y - 1)
    N[:, 2, 0] = 0.25 * xp * yp * (x + y - 1)
    N[:, 3, 0] = 0.25 * xm * yp * (y - x - 1)
    N[:, 4, 0] = 0.5 * (1 - x2) * ym
    N[:, 5, 0] = 0.5 * xp * (1 - y2)
    N[:, 6, 0] = 0.5 * (1 - x2) * yp
    N[:, 7, 0] = 0.5 * xm * (1 - y2)
    dN_dxi = np.zeros((n, 8, 2))
    dN_dxi[:, 0, 0] = 0.25 * ym * (2 * x + y)
    dN_dxi[:, 1, 0] = 0.25 * ym * (2 * x - y - 1)
    dN_dxi[:, 2, 0] = 0.25 * yp * (2 * x + y - 1)
    dN_dxi[:, 3, 0] = 0.25 * yp * (-2 * x + y - 1)
    dN_dxi[:, 4, 0] = -x * ym
    dN_dxi[:, 5, 0] = 0.5 * (1 - y2)
    dN_dxi[:, 6, 0] = -x * yp
    dN_dxi[:, 7, 0] = -0.5 * (1 - y2)
    dN_dxi[:, 0, 1] = 0.25 * xm * (2 * y + x)
    dN_dxi[:, 1, 1] = 0.25 * xp * (-2 * y + x - 1)
    dN_dxi[:, 2, 1] = 0.25 * xp * (2 * y + x - 1)
    dN_dxi[:, 3, 1] = 0.25 * xm * (2 * y - x - 1)
    dN_dxi[:, 4, 1] = -0.5 * (1 - x2)
    dN_dxi[:, 5, 1] = -y * xp
    dN_dxi[:, 6, 1] = 0.5 * (1 - x2)
    dN_dxi[:, 7, 1] = -y * xm
    return (N, dN_dxi)