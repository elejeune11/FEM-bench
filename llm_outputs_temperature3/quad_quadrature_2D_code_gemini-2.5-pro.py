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
        a = 1.0 / np.sqrt(3.0)
        p1d = np.array([-a, a])
        w1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        b = np.sqrt(3.0 / 5.0)
        p1d = np.array([-b, 0.0, b])
        w1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    else:
        raise ValueError('num_pts must be 1, 4, or 9')
    (xi, eta) = np.meshgrid(p1d, p1d)
    points = np.vstack([xi.ravel(), eta.ravel()]).T
    weights = np.outer(w1d, w1d).ravel()
    return (points, weights)