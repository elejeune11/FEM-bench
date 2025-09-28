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
        raise ValueError('Input must be a NumPy array')
    if xi.ndim == 1:
        if xi.shape != (2,):
            raise ValueError('1D input must have shape (2,)')
        xi = xi.reshape(1, 2)
    elif xi.ndim != 2 or xi.shape[1] != 2:
        raise ValueError('Input must have shape (n, 2)')
    if not np.all(np.isfinite(xi)):
        raise ValueError('Input contains non-finite values')
    ξ = xi[:, 0:1]
    η = xi[:, 1:2]
    ξm = 1 - ξ
    ξp = 1 + ξ
    ηm = 1 - η
    ηp = 1 + η
    ξ2 = 1 - ξ ** 2
    η2 = 1 - η ** 2
    N = np.zeros((len(xi), 8, 1))
    N[:, 0, 0] = -0.25 * ξm * ηm * (ξ + η + 1)
    N[:, 1, 0] = 0.25 * ξp * ηm * (ξ - η - 1)
    N[:, 2, 0] = 0.25 * ξp * ηp * (ξ + η - 1)
    N[:, 3, 0] = 0.25 * ξm * ηp * (η - ξ - 1)
    N[:, 4, 0] = 0.5 * ξ2 * ηm
    N[:, 5, 0] = 0.5 * ξp * η2
    N[:, 6, 0] = 0.5 * ξ2 * ηp
    N[:, 7, 0] = 0.5 * ξm * η2
    dN_dxi = np.zeros((len(xi), 8, 2))
    dN_dxi[:, 0, 0] = 0.25 * ηm * (2 * ξ + η)
    dN_dxi[:, 1, 0] = 0.25 * ηm * (2 * ξ - η - 1)
    dN_dxi[:, 2, 0] = 0.25 * ηp * (2 * ξ + η - 1)
    dN_dxi[:, 3, 0] = 0.25 * ηp * (-2 * ξ - η)
    dN_dxi[:, 4, 0] = -ξ * ηm
    dN_dxi[:, 5, 0] = 0.5 * η2
    dN_dxi[:, 6, 0] = -ξ * ηp
    dN_dxi[:, 7, 0] = -0.5 * η2
    dN_dxi[:, 0, 1] = 0.25 * ξm * (ξ + 2 * η)
    dN_dxi[:, 1, 1] = 0.25 * ξp * (-ξ + 2 * η)
    dN_dxi[:, 2, 1] = 0.25 * ξp * (ξ + 2 * η - 1)
    dN_dxi[:, 3, 1] = 0.25 * ξm * (2 * η - ξ - 1)
    dN_dxi[:, 4, 1] = -0.5 * ξ2
    dN_dxi[:, 5, 1] = -η * ξp
    dN_dxi[:, 6, 1] = 0.5 * ξ2
    dN_dxi[:, 7, 1] = -η * ξm
    return (N, dN_dxi)