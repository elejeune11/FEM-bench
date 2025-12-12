def FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.shape not in ((2,), (1, 2)) and len(xi.shape) != 2 or (len(xi.shape) == 2 and xi.shape[1] != 2):
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.isfinite(xi).all():
        raise ValueError('xi must contain only finite values')
    single_point = xi.shape == (2,)
    if single_point:
        xi = xi.reshape(1, 2)
    n = xi.shape[0]
    xi_vals = xi[:, 0]
    eta_vals = xi[:, 1]
    one_minus_xi = 1 - xi_vals
    one_plus_xi = 1 + xi_vals
    one_minus_eta = 1 - eta_vals
    one_plus_eta = 1 + eta_vals
    one_minus_xi2 = 1 - xi_vals ** 2
    one_minus_eta2 = 1 - eta_vals ** 2
    N = np.zeros((n, 8, 1))
    N[:, 0, 0] = -0.25 * one_minus_xi * one_minus_eta * (1 + xi_vals + eta_vals)
    N[:, 1, 0] = 0.25 * one_plus_xi * one_minus_eta * (xi_vals - eta_vals - 1)
    N[:, 2, 0] = 0.25 * one_plus_xi * one_plus_eta * (xi_vals + eta_vals - 1)
    N[:, 3, 0] = 0.25 * one_minus_xi * one_plus_eta * (eta_vals - xi_vals - 1)
    N[:, 4, 0] = 0.5 * one_minus_xi2 * one_minus_eta
    N[:, 5, 0] = 0.5 * one_plus_xi * one_minus_eta2
    N[:, 6, 0] = 0.5 * one_minus_xi2 * one_plus_eta
    N[:, 7, 0] = 0.5 * one_minus_xi * one_minus_eta2
    dN_dxi = np.zeros((n, 8, 2))
    dN1_dxi = 0.25 * one_minus_eta * (2 * xi_vals + eta_vals)
    dN1_deta = 0.25 * one_minus_xi * (xi_vals + 2 * eta_vals)
    dN_dxi[:, 0, 0] = dN1_dxi
    dN_dxi[:, 0, 1] = dN1_deta
    dN2_dxi = 0.25 * one_minus_eta * (2 * xi_vals - eta_vals)
    dN2_deta = -0.25 * one_plus_xi * (xi_vals - 2 * eta_vals - 1)
    dN_dxi[:, 1, 0] = dN2_dxi
    dN_dxi[:, 1, 1] = dN2_deta
    dN3_dxi = 0.25 * one_plus_eta * (2 * xi_vals + eta_vals)
    dN3_deta = 0.25 * one_plus_xi * (xi_vals + 2 * eta_vals - 1)
    dN_dxi[:, 2, 0] = dN3_dxi
    dN_dxi[:, 2, 1] = dN3_deta
    dN4_dxi = -0.25 * one_plus_eta * (2 * xi_vals - eta_vals)
    dN4_deta = 0.25 * one_minus_xi * (2 * eta_vals - xi_vals - 1)
    dN_dxi[:, 3, 0] = dN4_dxi
    dN_dxi[:, 3, 1] = dN4_deta
    dN5_dxi = -xi_vals * one_minus_eta
    dN5_deta = -0.5 * one_minus_xi2
    dN_dxi[:, 4, 0] = dN5_dxi
    dN_dxi[:, 4, 1] = dN5_deta
    dN6_dxi = 0.5 * one_minus_eta2
    dN6_deta = -eta_vals * one_plus_xi
    dN_dxi[:, 5, 0] = dN6_dxi
    dN_dxi[:, 5, 1] = dN6_deta
    dN7_dxi = -xi_vals * one_plus_eta
    dN7_deta = 0.5 * one_minus_xi2
    dN_dxi[:, 6, 0] = dN7_dxi
    dN_dxi[:, 6, 1] = dN7_deta
    dN8_dxi = -0.5 * one_minus_eta2
    dN8_deta = -eta_vals * one_minus_xi
    dN_dxi[:, 7, 0] = dN8_dxi
    dN_dxi[:, 7, 1] = dN8_deta
    if single_point:
        N = N.reshape(1, 8, 1)
        dN_dxi = dN_dxi.reshape(1, 8, 2)
    return (N, dN_dxi)