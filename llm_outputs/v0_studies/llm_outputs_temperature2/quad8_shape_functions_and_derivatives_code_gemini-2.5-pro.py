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
        raise ValueError('`xi` is not a NumPy array.')
    if xi.ndim == 1:
        if xi.shape == (2,):
            xi = xi.reshape(1, 2)
        else:
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    else:
        raise ValueError('`xi` has shape other than (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('`xi` contains non-finite values (NaN or Inf).')
    e = xi[:, 0]
    n = xi[:, 1]
    N1 = -0.25 * (1 - e) * (1 - n) * (1 + e + n)
    N2 = 0.25 * (1 + e) * (1 - n) * (e - n - 1)
    N3 = 0.25 * (1 + e) * (1 + n) * (e + n - 1)
    N4 = 0.25 * (1 - e) * (1 + n) * (n - e - 1)
    N5 = 0.5 * (1 - e ** 2) * (1 - n)
    N6 = 0.5 * (1 + e) * (1 - n ** 2)
    N7 = 0.5 * (1 - e ** 2) * (1 + n)
    N8 = 0.5 * (1 - e) * (1 - n ** 2)
    N = np.stack([N1, N2, N3, N4, N5, N6, N7, N8], axis=1)
    N = np.expand_dims(N, axis=2)
    dN1_de = 0.25 * (1 - n) * (2 * e + n)
    dN2_de = 0.25 * (1 - n) * (2 * e - n)
    dN3_de = 0.25 * (1 + n) * (2 * e + n)
    dN4_de = 0.25 * (1 + n) * (2 * e - n)
    dN5_de = -e * (1 - n)
    dN6_de = 0.5 * (1 - n ** 2)
    dN7_de = -e * (1 + n)
    dN8_de = -0.5 * (1 - n ** 2)
    dN_de = np.stack([dN1_de, dN2_de, dN3_de, dN4_de, dN5_de, dN6_de, dN7_de, dN8_de], axis=1)
    dN1_dn = 0.25 * (1 - e) * (2 * n + e)
    dN2_dn = 0.25 * (1 + e) * (2 * n - e)
    dN3_dn = 0.25 * (1 + e) * (2 * n + e)
    dN4_dn = 0.25 * (1 - e) * (2 * n - e)
    dN5_dn = -0.5 * (1 - e ** 2)
    dN6_dn = -n * (1 + e)
    dN7_dn = 0.5 * (1 - e ** 2)
    dN8_dn = -n * (1 - e)
    dN_dn = np.stack([dN1_dn, dN2_dn, dN3_dn, dN4_dn, dN5_dn, dN6_dn, dN7_dn, dN8_dn], axis=1)
    dN_dxi = np.stack([dN_de, dN_dn], axis=2)
    return (N, dN_dxi)