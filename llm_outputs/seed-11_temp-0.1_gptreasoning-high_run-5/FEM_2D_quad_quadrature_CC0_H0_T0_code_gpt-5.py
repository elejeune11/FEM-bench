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
    mapping = {1: 1, 4: 2, 9: 3}
    if num_pts not in mapping:
        raise ValueError('num_pts must be one of {1, 4, 9}.')
    n = mapping[num_pts]
    if n == 1:
        nodes = np.array([0.0], dtype=np.float64)
        w1d = np.array([2.0], dtype=np.float64)
    elif n == 2:
        a = 1.0 / np.sqrt(3.0)
        nodes = np.array([-a, a], dtype=np.float64)
        w1d = np.array([1.0, 1.0], dtype=np.float64)
    else:
        r = np.sqrt(3.0 / 5.0)
        nodes = np.array([-r, 0.0, r], dtype=np.float64)
        w1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    points = np.empty((num_pts, 2), dtype=np.float64)
    weights = np.empty((num_pts,), dtype=np.float64)
    k = 0
    for j in range(n):
        eta = nodes[j]
        wy = w1d[j]
        for i in range(n):
            xi = nodes[i]
            wx = w1d[i]
            points[k, 0] = xi
            points[k, 1] = eta
            weights[k] = wx * wy
            k += 1
    return (points, weights)