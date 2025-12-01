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
        xi_1d = np.array([0.0])
        w_1d = np.array([2.0])
    elif num_pts == 4:
        p = 1.0 / np.sqrt(3.0)
        xi_1d = np.array([-p, p])
        w_1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        p = np.sqrt(3.0 / 5.0)
        xi_1d = np.array([-p, 0.0, p])
        w_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    (xi, eta) = np.meshgrid(xi_1d, xi_1d)
    points = np.column_stack((xi.ravel(), eta.ravel()))
    (w_xi, w_eta) = np.meshgrid(w_1d, w_1d)
    weights = (w_xi * w_eta).ravel()
    return (points, weights)