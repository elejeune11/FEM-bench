def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (6-node) triangular shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates in the reference triangle.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 6, 1).
        Node order: [N1, N2, N3, N4, N5, N6].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t (ξ, η). Shape: (n, 6, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Uses P2 triangle with ξ_c = 1 - ξ - η:
        N1 = ξ(2ξ - 1),   N2 = η(2η - 1),   N3 = ξ_c(2ξ_c - 1),
        N4 = 4ξη,         N5 = 4ηξ_c,       N6 = 4ξξ_c.
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2).')
        xi = xi[None, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.isfinite(xi).all():
        raise ValueError('xi contains non-finite values (NaN or Inf).')
    n = xi.shape[0]
    xi1 = xi[:, 0]
    xi2 = xi[:, 1]
    xic = 1 - xi1 - xi2
    N = np.zeros((n, 6, 1))
    N[:, 0, 0] = xi1 * (2 * xi1 - 1)
    N[:, 1, 0] = xi2 * (2 * xi2 - 1)
    N[:, 2, 0] = xic * (2 * xic - 1)
    N[:, 3, 0] = 4 * xi1 * xi2
    N[:, 4, 0] = 4 * xi2 * xic
    N[:, 5, 0] = 4 * xi1 * xic
    dN_dxi = np.zeros((n, 6, 2))
    dN_dxi[:, 0, 0] = 4 * xi1 - 1
    dN_dxi[:, 0, 1] = 0
    dN_dxi[:, 1, 0] = 0
    dN_dxi[:, 1, 1] = 4 * xi2 - 1
    dN_dxi[:, 2, 0] = -4 * xic + 1
    dN_dxi[:, 2, 1] = -4 * xic + 1
    dN_dxi[:, 3, 0] = 4 * xi2
    dN_dxi[:, 3, 1] = 4 * xi1
    dN_dxi[:, 4, 0] = -4 * xi2
    dN_dxi[:, 4, 1] = 4 * xic - 4 * xi2
    dN_dxi[:, 5, 0] = 4 * xic - 4 * xi1
    dN_dxi[:, 5, 1] = -4 * xi1
    return (N, dN_dxi)