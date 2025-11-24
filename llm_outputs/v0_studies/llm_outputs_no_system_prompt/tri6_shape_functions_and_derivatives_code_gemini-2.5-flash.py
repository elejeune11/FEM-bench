def tri6_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (6-node) triangular shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates in the reference triangle.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 6, 1).
        Node order: [N1, N2, N3, N4, N5, N6].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t (ξ, η). Shape: (n, 6, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Uses P2 triangle with ξ_c = 1 - ξ - η:
        N1 = ξ(2ξ - 1),   N2 = η(2η - 1),   N3 = ξ_c(2ξ_c - 1),
        N4 = 4ξη,         N5 = 4ηξ_c,       N6 = 4ξξ_c.
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('Input `xi` must be a NumPy array.')
    original_shape = xi.shape
    if xi.ndim == 1:
        if original_shape != (2,):
            raise ValueError(f'Input `xi` has invalid shape {original_shape}. Expected (2,) or (n, 2).')
        xi_reshaped = xi[np.newaxis, :]
    elif xi.ndim == 2:
        if original_shape[1] != 2:
            raise ValueError(f'Input `xi` has invalid shape {original_shape}. Expected (2,) or (n, 2).')
        xi_reshaped = xi
    else:
        raise ValueError(f'Input `xi` has invalid shape {original_shape}. Expected (2,) or (n, 2).')
    if not np.all(np.isfinite(xi_reshaped)):
        raise ValueError('Input `xi` contains non-finite values (NaN or Inf).')
    n_points = xi_reshaped.shape[0]
    xi_coord = xi_reshaped[:, 0]
    eta_coord = xi_reshaped[:, 1]
    xi_c_coord = 1 - xi_coord - eta_coord
    N1 = xi_coord * (2 * xi_coord - 1)
    N2 = eta_coord * (2 * eta_coord - 1)
    N3 = xi_c_coord * (2 * xi_c_coord - 1)
    N4 = 4 * xi_coord * eta_coord
    N5 = 4 * eta_coord * xi_c_coord
    N6 = 4 * xi_coord * xi_c_coord
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=-1)
    N = N[:, :, np.newaxis]
    dN1_dxi = 4 * xi_coord - 1
    dN1_deta = np.zeros_like(xi_coord)
    dN2_dxi = np.zeros_like(xi_coord)
    dN2_deta = 4 * eta_coord - 1
    dN3_dxi = (4 * xi_c_coord - 1) * -1
    dN3_deta = (4 * xi_c_coord - 1) * -1
    dN4_dxi = 4 * eta_coord
    dN4_deta = 4 * xi_coord
    dN5_dxi = 4 * eta_coord * -1
    dN5_deta = 4 * xi_c_coord + 4 * eta_coord * -1
    dN6_dxi = 4 * xi_c_coord + 4 * xi_coord * -1
    dN6_deta = 4 * xi_coord * -1
    dN_dxi = np.zeros((n_points, 6, 2), dtype=xi_reshaped.dtype)
    dN_dxi[:, 0, 0] = dN1_dxi
    dN_dxi[:, 0, 1] = dN1_deta
    dN_dxi[:, 1, 0] = dN2_dxi
    dN_dxi[:, 1, 1] = dN2_deta
    dN_dxi[:, 2, 0] = dN3_dxi
    dN_dxi[:, 2, 1] = dN3_deta
    dN_dxi[:, 3, 0] = dN4_dxi
    dN_dxi[:, 3, 1] = dN4_deta
    dN_dxi[:, 4, 0] = dN5_dxi
    dN_dxi[:, 4, 1] = dN5_deta
    dN_dxi[:, 5, 0] = dN6_dxi
    dN_dxi[:, 5, 1] = dN6_deta
    return (N, dN_dxi)