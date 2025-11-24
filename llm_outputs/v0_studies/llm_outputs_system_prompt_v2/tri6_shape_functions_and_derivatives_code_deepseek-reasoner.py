def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('For 1D input, xi must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('For 2D input, xi must have shape (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.isfinite(xi).all():
        raise ValueError('xi contains non-finite values')
    n = xi.shape[0]
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    xi_c_vals = 1.0 - xi_vals - eta_vals
    N = np.zeros((n, 6, 1))
    N[:, 0, 0] = xi_vals * (2.0 * xi_vals - 1.0)
    N[:, 1, 0] = eta_vals * (2.0 * eta_vals - 1.0)
    N[:, 2, 0] = xi_c_vals * (2.0 * xi_c_vals - 1.0)
    N[:, 3, 0] = 4.0 * xi_vals * eta_vals
    N[:, 4, 0] = 4.0 * eta_vals * xi_c_vals
    N[:, 5, 0] = 4.0 * xi_vals * xi_c_vals
    dN_dxi = np.zeros((n, 6, 2))
    dN_dxi[:, 0, 0] = 4.0 * xi_vals - 1.0
    dN_dxi[:, 0, 1] = 0.0
    dN_dxi[:, 1, 0] = 0.0
    dN_dxi[:, 1, 1] = 4.0 * eta_vals - 1.0
    dN_dxi[:, 2, 0] = 4.0 * (xi_vals + eta_vals) - 3.0
    dN_dxi[:, 2, 1] = 4.0 * (xi_vals + eta_vals) - 3.0
    dN_dxi[:, 3, 0] = 4.0 * eta_vals
    dN_dxi[:, 3, 1] = 4.0 * xi_vals
    dN_dxi[:, 4, 0] = -4.0 * eta_vals
    dN_dxi[:, 4, 1] = 4.0 * (1.0 - xi_vals - 2.0 * eta_vals)
    dN_dxi[:, 5, 0] = 4.0 * (1.0 - 2.0 * xi_vals - eta_vals)
    dN_dxi[:, 5, 1] = -4.0 * xi_vals
    return (N, dN_dxi)