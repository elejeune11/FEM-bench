def FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.shape[-1] != 2:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values (NaN or Inf)')
    xi = xi.reshape(-1, 2)
    n = xi.shape[0]
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    one_minus_xi = 1 - xi_vals
    one_plus_xi = 1 + xi_vals
    one_minus_eta = 1 - eta_vals
    one_plus_eta = 1 + eta_vals
    xi_squared = xi_vals ** 2
    eta_squared = eta_vals ** 2
    N1 = -0.25 * one_minus_xi * one_minus_eta * (1 + xi_vals + eta_vals)
    N2 = 0.25 * one_plus_xi * one_minus_eta * (xi_vals - eta_vals - 1)
    N3 = 0.25 * one_plus_xi * one_plus_eta * (xi_vals + eta_vals - 1)
    N4 = 0.25 * one_minus_xi * one_plus_eta * (eta_vals - xi_vals - 1)
    N5 = 0.5 * (1 - xi_squared) * one_minus_eta
    N6 = 0.5 * one_plus_xi * (1 - eta_squared)
    N7 = 0.5 * (1 - xi_squared) * one_plus_eta
    N8 = 0.5 * one_minus_xi * (1 - eta_squared)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)
    N = N.reshape(n, 8, 1)
    dN1_dxi = -0.25 * (-one_minus_eta * (1 + xi_vals + eta_vals) + one_minus_xi * one_minus_eta)
    dN2_dxi = 0.25 * (one_minus_eta * (xi_vals - eta_vals - 1) + one_plus_xi * one_minus_eta)
    dN3_dxi = 0.25 * (one_plus_eta * (xi_vals + eta_vals - 1) + one_plus_xi * one_plus_eta)
    dN4_dxi = 0.25 * (-one_plus_eta * (eta_vals - xi_vals - 1) + one_minus_xi * one_plus_eta)
    dN5_dxi = 0.5 * (-2 * xi_vals) * one_minus_eta
    dN6_dxi = 0.5 * (1 - eta_squared)
    dN7_dxi = 0.5 * (-2 * xi_vals) * one_plus_eta
    dN8_dxi = 0.5 * -1 * (1 - eta_squared)
    dN1_deta = -0.25 * (one_minus_xi * (1 + xi_vals + eta_vals) + one_minus_xi * one_minus_eta)
    dN2_deta = 0.25 * (-one_plus_xi * (xi_vals - eta_vals - 1) + one_plus_xi * one_minus_eta)
    dN3_deta = 0.25 * (one_plus_xi * (xi_vals + eta_vals - 1) + one_plus_xi * one_plus_eta)
    dN4_deta = 0.25 * (one_minus_xi * (eta_vals - xi_vals - 1) + one_minus_xi * one_plus_eta)
    dN5_deta = 0.5 * (1 - xi_squared) * -1
    dN6_deta = 0.5 * one_plus_xi * (-2 * eta_vals)
    dN7_deta = 0.5 * (1 - xi_squared) * 1
    dN8_deta = 0.5 * one_minus_xi * (-2 * eta_vals)
    dN_dxi = np.stack([dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi, dN5_dxi, dN6_dxi, dN7_dxi, dN8_dxi], axis=1)
    dN_deta = np.stack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta], axis=1)
    dN_dxi = np.stack([dN_dxi, dN_deta], axis=2)
    return (N, dN_dxi)