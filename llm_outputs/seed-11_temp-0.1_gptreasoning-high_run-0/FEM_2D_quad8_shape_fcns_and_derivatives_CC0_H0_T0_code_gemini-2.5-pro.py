def FEM_2D_quad8_shape_fcns_and_derivatives_CC0_H0_T0(xi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    if xi.ndim == 1:
        if xi.shape == (2,):
            xi = xi.reshape(1, 2)
        else:
            raise ValueError('xi must have shape (2,) or (n, 2).')
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('xi must have shape (2,) or (n, 2).')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2).')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi must contain finite values.')
    e = xi[:, 0:1]
    n = xi[:, 1:2]
    e2 = e * e
    n2 = n * n
    one_m_e = 1.0 - e
    one_p_e = 1.0 + e
    one_m_n = 1.0 - n
    one_p_n = 1.0 + n
    one_m_e2 = 1.0 - e2
    one_m_n2 = 1.0 - n2
    N1 = -0.25 * one_m_e * one_m_n * (1.0 + e + n)
    N2 = 0.25 * one_p_e * one_m_n * (e - n - 1.0)
    N3 = 0.25 * one_p_e * one_p_n * (e + n - 1.0)
    N4 = 0.25 * one_m_e * one_p_n * (n - e - 1.0)
    N5 = 0.5 * one_m_e2 * one_m_n
    N6 = 0.5 * one_p_e * one_m_n2
    N7 = 0.5 * one_m_e2 * one_p_n
    N8 = 0.5 * one_m_e * one_m_n2
    N_matrix = np.hstack((N1, N2, N3, N4, N5, N6, N7, N8))
    N = N_matrix[:, :, np.newaxis]
    dN1_de = -0.25 * one_m_n * (-2.0 * e - n)
    dN2_de = 0.25 * one_m_n * (2.0 * e - n)
    dN3_de = 0.25 * one_p_n * (2.0 * e + n)
    dN4_de = 0.25 * one_p_n * (2.0 * e - n)
    dN5_de = -e * one_m_n
    dN6_de = 0.5 * one_m_n2
    dN7_de = -e * one_p_n
    dN8_de = -0.5 * one_m_n2
    dN1_dn = -0.25 * one_m_e * (-e - 2.0 * n)
    dN2_dn = 0.25 * one_p_e * (-e + 2.0 * n)
    dN3_dn = 0.25 * one_p_e * (e + 2.0 * n)
    dN4_dn = 0.25 * one_m_e * (2.0 * n - e)
    dN5_dn = -0.5 * one_m_e2
    dN6_dn = -n * one_p_e
    dN7_dn = 0.5 * one_m_e2
    dN8_dn = -n * one_m_e
    dN_de_matrix = np.hstack((dN1_de, dN2_de, dN3_de, dN4_de, dN5_de, dN6_de, dN7_de, dN8_de))
    dN_dn_matrix = np.hstack((dN1_dn, dN2_dn, dN3_dn, dN4_dn, dN5_dn, dN6_dn, dN7_dn, dN8_dn))
    dN_dxi = np.stack((dN_de_matrix, dN_dn_matrix), axis=2)
    return (N, dN_dxi)