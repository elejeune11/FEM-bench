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
    if num_pts not in {1, 4, 9}:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    if num_pts == 1:
        points = np.array([[0.0, 0.0]], dtype=np.float64)
        weights = np.array([4.0], dtype=np.float64)
    elif num_pts == 4:
        p = 1.0 / np.sqrt(3)
        points = np.array([[-p, -p], [-p, p], [p, -p], [p, p]], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    else:
        p = np.sqrt(0.6)
        w1 = 5.0 / 9.0
        w2 = 8.0 / 9.0
        points = np.array([[-p, -p], [-p, 0], [-p, p], [0, -p], [0, 0], [0, p], [p, -p], [p, 0], [p, p]], dtype=np.float64)
        weights = np.array([w1 * w1, w1 * w2, w1 * w1, w2 * w1, w2 * w2, w2 * w1, w1 * w1, w1 * w2, w1 * w1], dtype=np.float64)
    return (points, weights)