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
        pts_1d = np.array([0.0])
        wts_1d = np.array([2.0])
    elif num_pts == 4:
        pts_1d = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        wts_1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        pts_1d = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        wts_1d = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    (eta_vals, xi_vals) = np.meshgrid(pts_1d, pts_1d, indexing='ij')
    points = np.column_stack((xi_vals.ravel(order='C'), eta_vals.ravel(order='C')))
    weights = np.kron(wts_1d, wts_1d)
    return (points.astype(np.float64), weights.astype(np.float64))