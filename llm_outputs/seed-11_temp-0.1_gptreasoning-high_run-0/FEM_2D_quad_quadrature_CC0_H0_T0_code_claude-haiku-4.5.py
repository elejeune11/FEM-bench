def FEM_2D_quad_quadrature_CC0_H0_T0(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre quadrature points and weights for the
    reference square Q = { (xi, eta) : -1 ≤ xi ≤ 1, -1 ≤ eta ≤ 1 }.
    Supported rules (tensor products of 1D Gauss–Legendre):
    Parameters
    ----------
    num_pts : int
        Total number of quadrature points. Must be one of {1, 4, 9}.
    Returns
    -------
    points : (num_pts, 2) float64 ndarray
        Quadrature points [xi, eta] on the reference square.
        Points are returned in *row-major order* (η varies slowest, ξ varies fastest):
        for example, the 2×2 rule yields
            [(-1/√3, -1/√3), (+1/√3, -1/√3),
             (-1/√3, +1/√3), (+1/√3, +1/√3)].
    weights : (num_pts,) float64 ndarray
        Corresponding quadrature weights. Computed as the tensor product
        of 1D Gauss–Legendre weights; their sum equals 4.0, the area of Q.
    Raises
    ------
    ValueError
        If `num_pts` is not one of {1, 4, 9}.
    Notes
    -----
      of degree ≤(2n−1) in each variable.
    """
    if num_pts == 1:
        n = 1
    elif num_pts == 4:
        n = 2
    elif num_pts == 9:
        n = 3
    else:
        raise ValueError(f'num_pts must be one of {{1, 4, 9}}, got {num_pts}')
    (pts_1d, wts_1d) = np.polynomial.legendre.leggauss(n)
    points_list = []
    weights_list = []
    for eta_idx in range(n):
        for xi_idx in range(n):
            xi = pts_1d[xi_idx]
            eta = pts_1d[eta_idx]
            points_list.append([xi, eta])
            weight = wts_1d[xi_idx] * wts_1d[eta_idx]
            weights_list.append(weight)
    points = np.array(points_list, dtype=np.float64)
    weights = np.array(weights_list, dtype=np.float64)
    return (points, weights)