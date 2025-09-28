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
        pts_1d = np.array([0.0])
        wts_1d = np.array([2.0])
    elif num_pts == 4:
        p = 1.0 / np.sqrt(3.0)
        pts_1d = np.array([-p, p])
        wts_1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        p = np.sqrt(3.0 / 5.0)
        pts_1d = np.array([-p, 0.0, p])
        wts_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError(f'Unsupported number of quadrature points: {num_pts}. Must be one of {{1, 4, 9}}.')
    (xi, eta) = np.meshgrid(pts_1d, pts_1d)
    points = np.vstack([xi.ravel(), eta.ravel()]).T
    weights = np.outer(wts_1d, wts_1d).ravel()
    return (points, weights)