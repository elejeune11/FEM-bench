def compute_physical_gradient_quad8(node_coords: np.ndarray, node_values: np.ndarray, xi, eta) -> np.ndarray:
    coords = np.asarray(node_coords, dtype=float)
    if coords.shape != (8, 2):
        raise ValueError('node_coords must have shape (8,2).')
    if not np.all(np.isfinite(coords)):
        raise ValueError('node_coords must contain finite values.')
    values = np.asarray(node_values, dtype=float).ravel()
    if values.shape[0] != 8:
        raise ValueError('node_values must have length 8.')
    if not np.all(np.isfinite(values)):
        raise ValueError('node_values must contain finite values.')
    xi_arr = np.asarray(xi, dtype=float)
    eta_arr = np.asarray(eta, dtype=float)
    try:
        (xi_b, eta_b) = np.broadcast_arrays(xi_arr, eta_arr)
    except ValueError:
        raise ValueError('xi and eta must be broadcastable to the same shape.')
    if not (np.all(np.isfinite(xi_b)) and np.all(np.isfinite(eta_b))):
        raise ValueError('xi and eta must contain finite values.')
    xi_flat = xi_b.reshape(-1)
    eta_flat = eta_b.reshape(-1)
    pts = np.stack([xi_flat, eta_flat], axis=1)
    (_, dN) = quad8_shape_functions_and_derivatives(pts)
    J = np.einsum('ia,nib->nab', coords, dN)
    grad_nat = np.einsum('nib,i->nb', dN, values)
    a = J[:, 0, 0]
    b = J[:, 0, 1]
    c_ = J[:, 1, 0]
    d = J[:, 1, 1]
    det = a * d - b * c_
    if det.size and np.any(det == 0.0):
        raise ValueError('Jacobian is singular at one or more points.')
    invJT00 = d / det
    invJT01 = -c_ / det
    invJT10 = -b / det
    invJT11 = a / det
    gx = invJT00 * grad_nat[:, 0] + invJT01 * grad_nat[:, 1]
    gy = invJT10 * grad_nat[:, 0] + invJT11 * grad_nat[:, 1]
    grad = np.vstack((gx, gy))
    return grad