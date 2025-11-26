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
        raise ValueError('xi must be a NumPy array.')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain finite values (no NaN or Inf).')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError(f'xi with 1 dimension must have shape (2,), got {xi.shape}')
        xi_batch = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError(f'xi with 2 dimensions must have shape (n, 2), got {xi.shape}')
        xi_batch = xi
    else:
        raise ValueError(f'xi must be 1D or 2D, got {xi.ndim}D')
    r = xi_batch[:, 0]
    s = xi_batch[:, 1]
    one_minus_r = 1.0 - r
    one_plus_r = 1.0 + r
    one_minus_s = 1.0 - s
    one_plus_s = 1.0 + s
    r_sq = r * r
    s_sq = s * s
    one_minus_r_sq = 1.0 - r_sq
    one_minus_s_sq = 1.0 - s_sq
    N1 = -0.25 * one_minus_r * one_minus_s * (1.0 + r + s)
    N2 = 0.25 * one_plus_r * one_minus_s * (r - s - 1.0)
    N3 = 0.25 * one_plus_r * one_plus_s * (r + s - 1.0)
    N4 = 0.25 * one_minus_r * one_plus_s * (s - r - 1.0)
    N5 = 0.5 * one_minus_r_sq * one_minus_s
    N6 = 0.5 * one_plus_r * one_minus_s_sq
    N7 = 0.5 * one_minus_r_sq * one_plus_s
    N8 = 0.5 * one_minus_r * one_minus_s_sq
    N_stack = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)
    N = N_stack[:, :, np.newaxis]
    dN1_dr = 0.25 * one_minus_s * (2.0 * r + s)
    dN2_dr = 0.25 * one_minus_s * (2.0 * r - s)
    dN3_dr = 0.25 * one_plus_s * (2.0 * r + s)
    dN4_dr = 0.25 * one_plus_s * (2.0 * r - s)
    dN5_dr = -r * one_minus_s
    dN6_dr = 0.5 * one_minus_s_sq
    dN7_dr = -r * one_plus_s
    dN8_dr = -0.5 * one_minus_s_sq
    dN1_ds = 0.25 * one_minus_r * (r + 2.0 * s)
    dN2_ds = 0.25 * one_plus_r * (-r + 2.0 * s)
    dN3_ds = 0.25 * one_plus_r * (r + 2.0 * s)
    dN4_ds = 0.25 * one_minus_r * (-r + 2.0 * s)
    dN5_ds = -0.5 * one_minus_r_sq
    dN6_ds = -s * one_plus_r
    dN7_ds = 0.5 * one_minus_r_sq
    dN8_ds = -s * one_minus_r
    dN_dr = np.stack([dN1_dr, dN2_dr, dN3_dr, dN4_dr, dN5_dr, dN6_dr, dN7_dr, dN8_dr], axis=1)
    dN_ds = np.stack([dN1_ds, dN2_ds, dN3_ds, dN4_ds, dN5_ds, dN6_ds, dN7_ds, dN8_ds], axis=1)
    dN_dxi = np.stack([dN_dr, dN_ds], axis=2)
    return (N, dN_dxi)