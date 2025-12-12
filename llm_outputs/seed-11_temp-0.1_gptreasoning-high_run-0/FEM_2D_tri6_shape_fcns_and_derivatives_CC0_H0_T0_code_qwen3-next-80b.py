def FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.ndim not in [1, 2] or (xi.ndim == 1 and xi.shape[0] != 2) or (xi.ndim == 2 and xi.shape[1] != 2):
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values (NaN or Inf)')
    if xi.ndim == 1:
        xi = xi.reshape(1, 2)
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    xic_vals = 1 - xi_vals - eta_vals
    N1 = xi_vals * (2 * xi_vals - 1)
    N2 = eta_vals * (2 * eta_vals - 1)
    N3 = xic_vals * (2 * xic_vals - 1)
    N4 = 4 * xi_vals * eta_vals
    N5 = 4 * eta_vals * xic_vals
    N6 = 4 * xi_vals * xic_vals
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1).reshape(-1, 6, 1)
    dN1_dxi = 4 * xi_vals - 1
    dN1_deta = 0
    dN2_dxi = 0
    dN2_deta = 4 * eta_vals - 1
    dN3_dxi = -4 * xic_vals + 1
    dN3_deta = -4 * xic_vals + 1
    dN4_dxi = 4 * eta_vals
    dN4_deta = 4 * xi_vals
    dN5_dxi = -4 * eta_vals
    dN5_deta = 4 * (xic_vals - eta_vals)
    dN6_dxi = 4 * (xic_vals - xi_vals)
    dN6_deta = -4 * xi_vals
    dN_dxi = np.stack([np.stack([dN1_dxi, dN1_deta], axis=1), np.stack([dN2_dxi, dN2_deta], axis=1), np.stack([dN3_dxi, dN3_deta], axis=1), np.stack([dN4_dxi, dN4_deta], axis=1), np.stack([dN5_dxi, dN5_deta], axis=1), np.stack([dN6_dxi, dN6_deta], axis=1)], axis=1)
    return (N, dN_dxi)