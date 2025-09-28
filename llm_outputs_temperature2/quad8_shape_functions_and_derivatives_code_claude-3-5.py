def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (8-node) quadrilateral shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates (ξ, η) in the reference square.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 8, 1).
        Node order: [N1, N2, N3, N4, N5, N6, N7, N8].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t. (ξ, η). Shape: (n, 8, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('Input must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('1D input must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim != 2 or xi.shape[1] != 2:
        raise ValueError('Input must have shape (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input contains non-finite values')
    n = xi.shape[0]
    (x, y) = (xi[:, 0], xi[:, 1])
    xm = 1 - x
    xp = 1 + x
    ym = 1 - y
    yp = 1 + y
    x2 = 1 - x * x
    y2 = 1 - y * y
    N = np.zeros((n, 8, 1))
    N[:, 0, 0] = -0.25 * xm * ym * (1 + x + y)
    N[:, 1, 0] = 0.25 * xp * ym * (x - y - 1)
    N[:, 2, 0] = 0.25 * xp * yp * (x + y - 1)
    N[:, 3, 0] = 0.25 * xm * yp * (y - x - 1)
    N[:, 4, 0] = 0.5 * x2 * ym
    N[:, 5, 0] = 0.5 * xp * y2
    N[:, 6, 0] = 0.5 * x2 * yp
    N[:, 7, 0] = 0.5 * xm * y2
    dN_dxi = np.zeros((n, 8, 2))
    dN_dxi[:, 0, 0] = 0.25 * ym * (2 * x + y)
    dN_dxi[:, 1, 0] = 0.25 * ym * (2 * x - y - 1)
    dN_dxi[:, 2, 0] = 0.25 * yp * (2 * x + y - 1)
    dN_dxi[:, 3, 0] = 0.25 * yp * (-2 * x + y - 1)
    dN_dxi[:, 4, 0] = -x * ym
    dN_dxi[:, 5, 0] = 0.5 * y2
    dN_dxi[:, 6, 0] = -x * yp
    dN_dxi[:, 7, 0] = -0.5 * y2
    dN_dxi[:, 0, 1] = 0.25 * xm * (2 * y + x)
    dN_dxi[:, 1, 1] = 0.25 * xp * (-2 * y + x - 1)
    dN_dxi[:, 2, 1] = 0.25 * xp * (2 * y + x - 1)
    dN_dxi[:, 3, 1] = 0.25 * xm * (2 * y - x - 1)
    dN_dxi[:, 4, 1] = -0.5 * x2
    dN_dxi[:, 5, 1] = -xp * y
    dN_dxi[:, 6, 1] = 0.5 * x2
    dN_dxi[:, 7, 1] = -xm * y
    return (N, dN_dxi)