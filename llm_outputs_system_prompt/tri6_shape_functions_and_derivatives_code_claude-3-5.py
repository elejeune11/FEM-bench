def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if not isinstance(xi, np.ndarray):
        raise ValueError('Input must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('Single point must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim != 2 or xi.shape[1] != 2:
        raise ValueError('Input must have shape (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input contains non-finite values')
    n = xi.shape[0]
    xi_1 = xi[:, 0]
    xi_2 = xi[:, 1]
    xi_3 = 1 - xi_1 - xi_2
    N = np.zeros((n, 6, 1))
    dN_dxi = np.zeros((n, 6, 2))
    N[:, 0, 0] = xi_1 * (2 * xi_1 - 1)
    N[:, 1, 0] = xi_2 * (2 * xi_2 - 1)
    N[:, 2, 0] = xi_3 * (2 * xi_3 - 1)
    N[:, 3, 0] = 4 * xi_1 * xi_2
    N[:, 4, 0] = 4 * xi_2 * xi_3
    N[:, 5, 0] = 4 * xi_1 * xi_3
    dN_dxi[:, 0, 0] = 4 * xi_1 - 1
    dN_dxi[:, 1, 0] = 0
    dN_dxi[:, 2, 0] = -1 * (4 * xi_3 - 1)
    dN_dxi[:, 3, 0] = 4 * xi_2
    dN_dxi[:, 4, 0] = -4 * xi_2
    dN_dxi[:, 5, 0] = 4 * (xi_3 - xi_1)
    dN_dxi[:, 0, 1] = 0
    dN_dxi[:, 1, 1] = 4 * xi_2 - 1
    dN_dxi[:, 2, 1] = -1 * (4 * xi_3 - 1)
    dN_dxi[:, 3, 1] = 4 * xi_1
    dN_dxi[:, 4, 1] = 4 * (xi_3 - xi_2)
    dN_dxi[:, 5, 1] = -4 * xi_1
    return (N, dN_dxi)