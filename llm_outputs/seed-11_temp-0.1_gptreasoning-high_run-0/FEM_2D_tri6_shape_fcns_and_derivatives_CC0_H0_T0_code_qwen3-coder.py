def FEM_2D_tri6_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('xi must have shape (2,) or (n, 2)')
        xi = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain only finite values')
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    xi_c = 1 - xi_vals - eta_vals
    N1 = xi_vals * (2 * xi_vals - 1)
    N2 = eta_vals * (2 * eta_vals - 1)
    N3 = xi_c * (2 * xi_c - 1)
    N4 = 4 * xi_vals * eta_vals
    N5 = 4 * eta_vals * xi_c
    N6 = 4 * xi_vals * xi_c
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1)
    N = N[:, :, np.newaxis]
    dN1_dxi = 4 * xi_vals - 1
    dN1_deta = np.zeros_like(xi_vals)
    dN2_dxi = np.zeros_like(xi_vals)
    dN2_deta = 4 * eta_vals - 1
    dN3_dxi = -4 * xi_c + 1
    dN3_deta = -4 * xi_c + 1
    dN4_dxi = 4 * eta_vals
    dN4_deta = 4 * xi_vals
    dN5_dxi = -4 * eta_vals
    dN5_deta = 4 * (xi_c - eta_vals)
    dN6_dxi = 4 * (xi_c - xi_vals)
    dN6_deta = -4 * xi_vals
    dN_dxi = np.stack([np.stack([dN1_dxi, dN1_deta], axis=1), np.stack([dN2_dxi, dN2_deta], axis=1), np.stack([dN3_dxi, dN3_deta], axis=1), np.stack([dN4_dxi, dN4_deta], axis=1), np.stack([dN5_dxi, dN5_deta], axis=1), np.stack([dN6_dxi, dN6_deta], axis=1)], axis=1)
    return (N, dN_dxi)