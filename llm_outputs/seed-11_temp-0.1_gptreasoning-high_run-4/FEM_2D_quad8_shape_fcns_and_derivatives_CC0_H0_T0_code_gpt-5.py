def FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        raise ValueError('`xi` must be a NumPy array.')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        x = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
        x = xi
    else:
        raise ValueError('`xi` must have shape (2,) for a single point or (n, 2) for multiple points.')
    if not np.all(np.isfinite(x)):
        raise ValueError('`xi` must contain only finite values (no NaN or Inf).')
    r = x[:, 0]
    s = x[:, 1]
    N1 = -0.25 * (1.0 - r) * (1.0 - s) * (1.0 + r + s)
    N2 = 0.25 * (1.0 + r) * (1.0 - s) * (r - s - 1.0)
    N3 = 0.25 * (1.0 + r) * (1.0 + s) * (r + s - 1.0)
    N4 = 0.25 * (1.0 - r) * (1.0 + s) * (s - r - 1.0)
    N5 = 0.5 * (1.0 - r * r) * (1.0 - s)
    N6 = 0.5 * (1.0 + r) * (1.0 - s * s)
    N7 = 0.5 * (1.0 - r * r) * (1.0 + s)
    N8 = 0.5 * (1.0 - r) * (1.0 - s * s)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[..., np.newaxis]
    dNdr1 = 0.25 * (1.0 - s) * (2.0 * r + s)
    dNdS1 = 0.25 * (1.0 - r) * (r + 2.0 * s)
    dNdr2 = 0.25 * (1.0 - s) * (2.0 * r - s)
    dNdS2 = -0.25 * (1.0 + r) * (r - 2.0 * s)
    dNdr3 = 0.25 * (1.0 + s) * (2.0 * r + s)
    dNdS3 = 0.25 * (1.0 + r) * (r + 2.0 * s)
    dNdr4 = 0.25 * (1.0 + s) * (2.0 * r - s)
    dNdS4 = 0.25 * (1.0 - r) * (2.0 * s - r)
    dNdr5 = -r * (1.0 - s)
    dNdS5 = -0.5 * (1.0 - r * r)
    dNdr6 = 0.5 * (1.0 - s * s)
    dNdS6 = -(1.0 + r) * s
    dNdr7 = -r * (1.0 + s)
    dNdS7 = 0.5 * (1.0 - r * r)
    dNdr8 = -0.5 * (1.0 - s * s)
    dNdS8 = -(1.0 - r) * s
    dNdr = np.stack([dNdr1, dNdr2, dNdr3, dNdr4, dNdr5, dNdr6, dNdr7, dNdr8], axis=1)
    dNdS = np.stack([dNdS1, dNdS2, dNdS3, dNdS4, dNdS5, dNdS6, dNdS7, dNdS8], axis=1)
    dN_dxi = np.stack([dNdr, dNdS], axis=2)
    return (N, dN_dxi)