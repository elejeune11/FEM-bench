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
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        x = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
        x = xi
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    x = x.astype(float, copy=False)
    if not np.all(np.isfinite(x)):
        raise ValueError('xi must contain only finite values.')
    r = x[:, 0]
    s = x[:, 1]
    N1 = -0.25 * (1 - r) * (1 - s) * (1 + r + s)
    N2 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
    N3 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
    N4 = 0.25 * (1 - r) * (1 + s) * (s - r - 1)
    N5 = 0.5 * (1 - r ** 2) * (1 - s)
    N6 = 0.5 * (1 + r) * (1 - s ** 2)
    N7 = 0.5 * (1 - r ** 2) * (1 + s)
    N8 = 0.5 * (1 - r) * (1 - s ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[:, :, None]
    dNdr1 = 0.25 * (1 - s) * (2 * r + s)
    dNds1 = 0.25 * (1 - r) * (r + 2 * s)
    dNdr2 = 0.25 * (1 - s) * (2 * r - s)
    dNds2 = -0.25 * (1 + r) * (r - 2 * s)
    dNdr3 = 0.25 * (1 + s) * (2 * r + s)
    dNds3 = 0.25 * (1 + r) * (r + 2 * s)
    dNdr4 = 0.25 * (1 + s) * (2 * r - s)
    dNds4 = 0.25 * (1 - r) * (2 * s - r)
    dNdr5 = -r * (1 - s)
    dNds5 = -0.5 * (1 - r ** 2)
    dNdr6 = 0.5 * (1 - s ** 2)
    dNds6 = -s * (1 + r)
    dNdr7 = -r * (1 + s)
    dNds7 = 0.5 * (1 - r ** 2)
    dNdr8 = -0.5 * (1 - s ** 2)
    dNds8 = -s * (1 - r)
    dNdr = np.stack([dNdr1, dNdr2, dNdr3, dNdr4, dNdr5, dNdr6, dNdr7, dNdr8], axis=1)
    dNds = np.stack([dNds1, dNds2, dNds3, dNds4, dNds5, dNds6, dNds7, dNds8], axis=1)
    dN_dxi = np.stack([dNdr, dNds], axis=2)
    return (N, dN_dxi)