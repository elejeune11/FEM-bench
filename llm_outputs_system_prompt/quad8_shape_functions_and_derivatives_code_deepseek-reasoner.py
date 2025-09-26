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
        raise ValueError('xi must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('For 1D input, xi must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim == 2:
        if xi.shape[1] != 2:
            raise ValueError('For 2D input, xi must have shape (n, 2)')
    else:
        raise ValueError('xi must have shape (2,) or (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('xi contains non-finite values')
    xi = xi.reshape(-1, 2)
    n_points = xi.shape[0]
    ξ = xi[:, 0]
    η = xi[:, 1]
    ξ2 = ξ ** 2
    η2 = η ** 2
    one_minus_ξ = 1 - ξ
    one_plus_ξ = 1 + ξ
    one_minus_η = 1 - η
    one_plus_η = 1 + η
    N1 = -0.25 * one_minus_ξ * one_minus_η * (1 + ξ + η)
    N2 = 0.25 * one_plus_ξ * one_minus_η * (ξ - η - 1)
    N3 = 0.25 * one_plus_ξ * one_plus_η * (ξ + η - 1)
    N4 = 0.25 * one_minus_ξ * one_plus_η * (η - ξ - 1)
    N5 = 0.5 * (1 - ξ2) * one_minus_η
    N6 = 0.5 * one_plus_ξ * (1 - η2)
    N7 = 0.5 * (1 - ξ2) * one_plus_η
    N8 = 0.5 * one_minus_ξ * (1 - η2)
    N = np.column_stack([N1, N2, N3, N4, N5, N6, N7, N8]).reshape(n_points, 8, 1)
    dN1_dξ = -0.25 * (-one_minus_η * (1 + ξ + η) + one_minus_ξ * one_minus_η)
    dN2_dξ = 0.25 * (one_minus_η * (ξ - η - 1) + one_plus_ξ * one_minus_η)
    dN3_dξ = 0.25 * (one_plus_η * (ξ + η - 1) + one_plus_ξ * one_plus_η)
    dN4_dξ = 0.25 * (-one_plus_η * (η - ξ - 1) - one_minus_ξ * one_plus_η)
    dN5_dξ = 0.5 * (-2 * ξ * one_minus_η)
    dN6_dξ = 0.5 * (1 - η2)
    dN7_dξ = 0.5 * (-2 * ξ * one_plus_η)
    dN8_dξ = 0.5 * -(1 - η2)
    dN1_dη = -0.25 * (-one_minus_ξ * (1 + ξ + η) + one_minus_ξ * one_minus_η)
    dN2_dη = 0.25 * (-one_plus_ξ * (ξ - η - 1) - one_plus_ξ * one_minus_η)
    dN3_dη = 0.25 * (one_plus_ξ * (ξ + η - 1) + one_plus_ξ * one_plus_η)
    dN4_dη = 0.25 * (one_minus_ξ * (η - ξ - 1) + one_minus_ξ * one_plus_η)
    dN5_dη = 0.5 * -(1 - ξ2)
    dN6_dη = 0.5 * one_plus_ξ * (-2 * η)
    dN7_dη = 0.5 * (1 - ξ2)
    dN8_dη = 0.5 * one_minus_ξ * (-2 * η)
    dN_dξ = np.column_stack([dN1_dξ, dN2_dξ, dN3_dξ, dN4_dξ, dN5_dξ, dN6_dξ, dN7_dξ, dN8_dξ])
    dN_dη = np.column_stack([dN1_dη, dN2_dη, dN3_dη, dN4_dη, dN5_dη, dN6_dη, dN7_dη, dN8_dη])
    dN_dxi = np.stack([dN_dξ, dN_dη], axis=-1)
    return (N, dN_dxi)