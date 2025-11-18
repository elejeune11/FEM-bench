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
        xi = np.array([0.0])
        w = np.array([2.0])
    elif num_pts == 4:
        xi = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
        w = np.array([1.0, 1.0])
    elif num_pts == 9:
        xi = np.array([-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])
        w = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('num_pts must be 1, 4, or 9')
    points = np.array([[x, y] for x in xi for y in xi])
    weights = np.array([wx * wy for wx in w for wy in w])
    return (points, weights)