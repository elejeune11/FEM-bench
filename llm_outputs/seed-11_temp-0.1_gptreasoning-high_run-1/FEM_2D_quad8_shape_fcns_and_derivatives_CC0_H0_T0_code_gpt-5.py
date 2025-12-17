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
            raise ValueError('`xi` must have shape (2,) or (n, 2).')
        pts = xi.reshape(1, 2)
    elif xi.ndim == 2 and xi.shape[1] == 2:
        pts = xi
    else:
        raise ValueError('`xi` must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(pts)):
        raise ValueError('`xi` must contain only finite values.')
    r = pts[:, 0]
    s = pts[:, 1]
    N1 = -0.25 * (1 - r) * (1 - s) * (1 + r + s)
    N2 = 0.25 * (1 + r) * (1 - s) * (r - s - 1)
    N3 = 0.25 * (1 + r) * (1 + s) * (r + s - 1)
    N4 = 0.25 * (1 - r) * (1 + s) * (s - r - 1)
    N5 = 0.5 * (1 - r ** 2) * (1 - s)
    N6 = 0.5 * (1 + r) * (1 - s ** 2)
    N7 = 0.5 * (1 - r ** 2) * (1 + s)
    N8 = 0.5 * (1 - r) * (1 - s ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)[..., None]
    dN1_dr = 0.25 * (1 - s) * (2 * r + s)
    dN2_dr = 0.25 * (1 - s) * (2 * r - s)
    dN3_dr = 0.25 * (1 + s) * (2 * r + s)
    dN4_dr = 0.25 * (1 + s) * (2 * r - s)
    dN5_dr = -r * (1 - s)
    dN6_dr = 0.5 * (1 - s ** 2)
    dN7_dr = -r * (1 + s)
    dN8_dr = -0.5 * (1 - s ** 2)
    dN1_ds = 0.25 * (1 - r) * (r + 2 * s)
    dN2_ds = -0.25 * (1 + r) * (r - 2 * s)
    dN3_ds = 0.25 * (1 + r) * (r + 2 * s)
    dN4_ds = 0.25 * (1 - r) * (2 * s - r)
    dN5_ds = -0.5 * (1 - r ** 2)
    dN6_ds = -(1 + r) * s
    dN7_ds = 0.5 * (1 - r ** 2)
    dN8_ds = -(1 - r) * s
    dN_dr = np.stack([dN1_dr, dN2_dr, dN3_dr, dN4_dr, dN5_dr, dN6_dr, dN7_dr, dN8_dr], axis=1)
    dN_ds = np.stack([dN1_ds, dN2_ds, dN3_ds, dN4_ds, dN5_ds, dN6_ds, dN7_ds, dN8_ds], axis=1)
    dN_dxi = np.stack([dN_dr, dN_ds], axis=2)
    return (N, dN_dxi)