def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape[0] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values')
    n = xi.shape[0]
    xi0 = xi[:, 0]
    xi1 = xi[:, 1]
    N = np.zeros((n, 8, 1))
    dN_dxi = np.zeros((n, 8, 2))
    N[:, 0, 0] = -0.25 * (1 - xi0) * (1 - xi1) * (1 + xi0 + xi1)
    dN_dxi[:, 0, 0] = -0.25 * (-(1 - xi1) * (1 + xi0 + xi1) + (1 - xi0) * (1 - xi1) * 1)
    dN_dxi[:, 0, 1] = -0.25 * (-(1 - xi0) * (1 + xi0 + xi1) + (1 - xi0) * (1 - xi1) * 1)
    N[:, 1, 0] = 0.25 * (1 + xi0) * (1 - xi1) * (xi0 - xi1 - 1)
    dN_dxi[:, 1, 0] = 0.25 * ((1 - xi1) * (xi0 - xi1 - 1) + (1 + xi0) * (1 - xi1) * 1)
    dN_dxi[:, 1, 1] = 0.25 * (-(1 + xi0) * (xi0 - xi1 - 1) + (1 + xi0) * (1 - xi1) * -1)
    N[:, 2, 0] = 0.25 * (1 + xi0) * (1 + xi1) * (xi0 + xi1 - 1)
    dN_dxi[:, 2, 0] = 0.25 * ((1 + xi1) * (xi0 + xi1 - 1) + (1 + xi0) * (1 + xi1) * 1)
    dN_dxi[:, 2, 1] = 0.25 * ((1 + xi0) * (xi0 + xi1 - 1) + (1 + xi0) * (1 + xi1) * 1)
    N[:, 3, 0] = 0.25 * (1 - xi0) * (1 + xi1) * (xi1 - xi0 - 1)
    dN_dxi[:, 3, 0] = 0.25 * (-(1 + xi1) * (xi1 - xi0 - 1) + (1 - xi0) * (1 + xi1) * -1)
    dN_dxi[:, 3, 1] = 0.25 * ((1 - xi0) * (xi1 - xi0 - 1) + (1 - xi0) * (1 + xi1) * 1)
    N[:, 4, 0] = 0.5 * (1 - xi0 ** 2) * (1 - xi1)
    dN_dxi[:, 4, 0] = 0.5 * (-2 * xi0) * (1 - xi1)
    dN_dxi[:, 4, 1] = 0.5 * (1 - xi0 ** 2) * -1
    N[:, 5, 0] = 0.5 * (1 + xi0) * (1 - xi1 ** 2)
    dN_dxi[:, 5, 0] = 0.5 * 1 * (1 - xi1 ** 2)
    dN_dxi[:, 5, 1] = 0.5 * (1 + xi0) * (-2 * xi1)
    N[:, 6, 0] = 0.5 * (1 - xi0 ** 2) * (1 + xi1)
    dN_dxi[:, 6, 0] = 0.5 * (-2 * xi0) * (1 + xi1)
    dN_dxi[:, 6, 1] = 0.5 * (1 - xi0 ** 2) * 1
    N[:, 7, 0] = 0.5 * (1 - xi0) * (1 - xi1 ** 2)
    dN_dxi[:, 7, 0] = 0.5 * -1 * (1 - xi1 ** 2)
    dN_dxi[:, 7, 1] = 0.5 * (1 - xi0) * (-2 * xi1)
    return (N, dN_dxi)