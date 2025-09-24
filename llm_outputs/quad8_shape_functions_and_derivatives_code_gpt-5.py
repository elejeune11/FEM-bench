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
    import numpy as np
    if not isinstance(xi, np.ndarray):
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        Xi = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        Xi = xi
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(Xi)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    x = Xi[:, 0]
    y = Xi[:, 1]
    one_minus_x = 1.0 - x
    one_plus_x = 1.0 + x
    one_minus_y = 1.0 - y
    one_plus_y = 1.0 + y
    one_minus_x2 = 1.0 - x * x
    one_minus_y2 = 1.0 - y * y
    N1 = -0.25 * one_minus_x * one_minus_y * (1.0 + x + y)
    N2 = 0.25 * one_plus_x * one_minus_y * (x - y - 1.0)
    N3 = 0.25 * one_plus_x * one_plus_y * (x + y - 1.0)
    N4 = 0.25 * one_minus_x * one_plus_y * (y - x - 1.0)
    N5 = 0.5 * one_minus_x2 * one_minus_y
    N6 = 0.5 * one_plus_x * one_minus_y2
    N7 = 0.5 * one_minus_x2 * one_plus_y
    N8 = 0.5 * one_minus_x * one_minus_y2
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[..., np.newaxis]
    dN1_dxi = 0.25 * one_minus_y * (2.0 * x + y)
    dN1_deta = 0.25 * one_minus_x * (x + 2.0 * y)
    dN2_dxi = 0.25 * one_minus_y * (2.0 * x - y)
    dN2_deta = -0.25 * one_plus_x * (x - 2.0 * y)
    dN3_dxi = 0.25 * one_plus_y * (2.0 * x + y)
    dN3_deta = 0.25 * one_plus_x * (x + 2.0 * y)
    dN4_dxi = 0.25 * one_plus_y * (2.0 * x - y)
    dN4_deta = 0.25 * one_minus_x * (2.0 * y - x)
    dN5_dxi = -x * one_minus_y
    dN5_deta = -0.5 * one_minus_x2
    dN6_dxi = 0.5 * one_minus_y2
    dN6_deta = -y * one_plus_x
    dN7_dxi = -x * one_plus_y
    dN7_deta = 0.5 * one_minus_x2
    dN8_dxi = -0.5 * one_minus_y2
    dN8_deta = -y * one_minus_x
    n = Xi.shape[0]
    dN_dxi = np.empty((n, 8, 2), dtype=N.dtype)
    dN_dxi[:, 0, 0] = dN1_dxi
    dN_dxi[:, 0, 1] = dN1_deta
    dN_dxi[:, 1, 0] = dN2_dxi
    dN_dxi[:, 1, 1] = dN2_deta
    dN_dxi[:, 2, 0] = dN3_dxi
    dN_dxi[:, 2, 1] = dN3_deta
    dN_dxi[:, 3, 0] = dN4_dxi
    dN_dxi[:, 3, 1] = dN4_deta
    dN_dxi[:, 4, 0] = dN5_dxi
    dN_dxi[:, 4, 1] = dN5_deta
    dN_dxi[:, 5, 0] = dN6_dxi
    dN_dxi[:, 5, 1] = dN6_deta
    dN_dxi[:, 6, 0] = dN7_dxi
    dN_dxi[:, 6, 1] = dN7_deta
    dN_dxi[:, 7, 0] = dN8_dxi
    dN_dxi[:, 7, 1] = dN8_deta
    return (N, dN_dxi)