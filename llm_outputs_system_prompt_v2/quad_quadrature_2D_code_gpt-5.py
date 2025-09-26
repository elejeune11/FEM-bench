def quad_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre quadrature points and weights for the
    reference square Q = { (xi, eta) : -1 <= xi <= 1, -1 <= eta <= 1 }.
    Supported rules (tensor products of 1D Gauss–Legendre):
    Parameters
    ----------
    num_pts : int
        Total number of quadrature points (1, 4, or 9).
    Returns
    -------
    points : (num_pts, 2) float64 ndarray
        Quadrature points [xi, eta] on the reference square.
    weights : (num_pts,) float64 ndarray
        Quadrature weights corresponding to `points`. The sum of weights
        equals the area of Q, which is 4.0.
    Raises
    ------
    ValueError
        If `num_pts` is not one of {1, 4, 9}.
    """
    if num_pts == 1:
        x1d = np.array([0.0], dtype=np.float64)
        w1d = np.array([2.0], dtype=np.float64)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        x1d = np.array([-a, a], dtype=np.float64)
        w1d = np.array([1.0, 1.0], dtype=np.float64)
    elif num_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        x1d = np.array([-a, 0.0, a], dtype=np.float64)
        w1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    n = x1d.size
    points = np.empty((n * n, 2), dtype=np.float64)
    weights = np.empty(n * n, dtype=np.float64)
    k = 0
    for j in range(n):
        eta = x1d[j]
        w_eta = w1d[j]
        for i in range(n):
            xi = x1d[i]
            w_xi = w1d[i]
            points[k, 0] = xi
            points[k, 1] = eta
            weights[k] = w_xi * w_eta
            k += 1
    return (points, weights)