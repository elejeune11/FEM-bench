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
        p1d = np.array([0.0])
        w1d = np.array([2.0])
    elif num_pts == 4:
        p1d = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        w1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        p1d = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        w1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('Unsupported number of quadrature points. Must be one of {1, 4, 9}.')
    (xi, eta) = np.meshgrid(p1d, p1d)
    (w_xi, w_eta) = np.meshgrid(w1d, w1d)
    points = np.vstack([xi.ravel(), eta.ravel()]).T
    weights = (w_xi * w_eta).ravel()
    return (points, weights)