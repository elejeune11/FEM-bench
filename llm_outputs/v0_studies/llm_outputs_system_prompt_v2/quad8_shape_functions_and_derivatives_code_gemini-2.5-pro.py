def quad8_shape_functions_and_derivatives(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation of quadratic (8-node) quadrilateral shape functions and derivatives.
    Parameters
    ----------
    xi : np.ndarray
        Natural coordinates (ξ, η) in the reference square.
    Returns
    -------
    N : np.ndarray
        Shape functions evaluated at the input points. Shape: (n, 8, 1).
        Node order: [N1, N2, N3, N4, N5, N6, N7, N8].
    dN_dxi : np.ndarray
        Partial derivatives w.r.t. (ξ, η). Shape: (n, 8, 2).
        Columns correspond to [∂()/∂ξ, ∂()/∂η] in the same node order.
    Raises
    ------
    ValueError
        If `xi` is not a NumPy array.
        If `xi` has shape other than (2,) or (n, 2).
        If `xi` contains non-finite values (NaN or Inf).
    Notes
    -----
    Shape functions:
        N1 = -1/4 (1-ξ)(1-η)(1+ξ+η)
        N2 =  +1/4 (1+ξ)(1-η)(ξ-η-1)
        N3 =  +1/4 (1+ξ)(1+η)(ξ+η-1)
        N4 =  +1/4 (1-ξ)(1+η)(η-ξ-1)
        N5 =  +1/2 (1-ξ²)(1-η)
        N6 =  +1/2 (1+ξ)(1-η²)
        N7 =  +1/2 (1-ξ²)(1+η)
        N8 =  +1/2 (1-ξ)(1-η²)
    """
    if not isinstance(xi, np.ndarray):
        raise ValueError('xi must be a NumPy array.')
    if xi.ndim not in [1, 2] or xi.shape[-1] != 2:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain finite values.')
    xi_2d = np.atleast_2d(xi)
    ksi = xi_2d[:, 0:1]
    eta = xi_2d[:, 1:2]
    ksi2 = ksi * ksi
    eta2 = eta * eta
    N1 = -0.25 * (1 - ksi) * (1 - eta) * (1 + ksi + eta)
    N2 = 0.25 * (1 + ksi) * (1 - eta) * (ksi - eta - 1)
    N3 = 0.25 * (1 + ksi) * (1 + eta) * (ksi + eta - 1)
    N4 = 0.25 * (1 - ksi) * (1 + eta) * (eta - ksi - 1)
    N5 = 0.5 * (1 - ksi2) * (1 - eta)
    N6 = 0.5 * (1 + ksi) * (1 - eta2)
    N7 = 0.5 * (1 - ksi2) * (1 + eta)
    N8 = 0.5 * (1 - ksi) * (1 - eta2)
    N = np.hstack([N1, N2, N3, N4, N5, N6, N7, N8])
    N = N[:, :, np.newaxis]
    dN1_dksi = 0.25 * (1 - eta) * (2 * ksi + eta)
    dN2_dksi = 0.25 * (1 - eta) * (2 * ksi - eta)
    dN3_dksi = 0.25 * (1 + eta) * (2 * ksi + eta)
    dN4_dksi = 0.25 * (1 + eta) * (2 * ksi - eta)
    dN5_dksi = -ksi * (1 - eta)
    dN6_dksi = 0.5 * (1 - eta2)
    dN7_dksi = -ksi * (1 + eta)
    dN8_dksi = -0.5 * (1 - eta2)
    dN_dksi = np.hstack([dN1_dksi, dN2_dksi, dN3_dksi, dN4_dksi, dN5_dksi, dN6_dksi, dN7_dksi, dN8_dksi])
    dN1_deta = 0.25 * (1 - ksi) * (ksi + 2 * eta)
    dN2_deta = 0.25 * (1 + ksi) * (2 * eta - ksi)
    dN3_deta = 0.25 * (1 + ksi) * (ksi + 2 * eta)
    dN4_deta = 0.25 * (1 - ksi) * (2 * eta - ksi)
    dN5_deta = -0.5 * (1 - ksi2)
    dN6_deta = -eta * (1 + ksi)
    dN7_deta = 0.5 * (1 - ksi2)
    dN8_deta = -eta * (1 - ksi)
    dN_deta = np.hstack([dN1_deta, dN2_deta, dN3_deta, dN4_deta, dN5_deta, dN6_deta, dN7_deta, dN8_deta])
    dN_dxi = np.stack([dN_dksi, dN_deta], axis=2)
    return (N, dN_dxi)