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
        raise ValueError('Input must be a NumPy array')
    if xi.shape == (2,):
        xi = xi.reshape(1, 2)
    elif len(xi.shape) != 2 or xi.shape[1] != 2:
        raise ValueError('Input must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input contains non-finite values')
    n = xi.shape[0]
    (x, y) = (xi[:, 0], xi[:, 1])
    z = 1 - x - y
    N1 = x * (2 * x - 1)
    N2 = y * (2 * y - 1)
    N3 = z * (2 * z - 1)
    N4 = 4 * x * y
    N5 = 4 * y * z
    N6 = 4 * x * z
    N = np.stack([N1, N2, N3, N4, N5, N6], axis=1).reshape(n, 6, 1)
    dN1_dx = 4 * x - 1
    dN1_dy = 0
    dN2_dx = 0
    dN2_dy = 4 * y - 1
    dN3_dx = -1 * (4 * z - 1)
    dN3_dy = -1 * (4 * z - 1)
    dN4_dx = 4 * y
    dN4_dy = 4 * x
    dN5_dx = -4 * y
    dN5_dy = 4 * (z - y)
    dN6_dx = 4 * (z - x)
    dN6_dy = -4 * x
    dN_dxi = np.stack([[dN1_dx, dN1_dy], [dN2_dx, dN2_dy], [dN3_dx, dN3_dy], [dN4_dx, dN4_dy], [dN5_dx, dN5_dy], [dN6_dx, dN6_dy]], axis=1)
    dN_dxi = np.tile(dN_dxi, (n, 1, 1))
    return (N, dN_dxi)