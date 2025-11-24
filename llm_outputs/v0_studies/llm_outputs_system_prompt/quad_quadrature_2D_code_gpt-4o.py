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
        points = np.array([[0.0, 0.0]])
        weights = np.array([4.0])
    elif num_pts == 4:
        sqrt_1_3 = np.sqrt(1 / 3)
        points = np.array([[-sqrt_1_3, -sqrt_1_3], [sqrt_1_3, -sqrt_1_3], [-sqrt_1_3, sqrt_1_3], [sqrt_1_3, sqrt_1_3]])
        weights = np.array([1.0, 1.0, 1.0, 1.0])
    elif num_pts == 9:
        sqrt_3_5 = np.sqrt(3 / 5)
        points = np.array([[-sqrt_3_5, -sqrt_3_5], [0.0, -sqrt_3_5], [sqrt_3_5, -sqrt_3_5], [-sqrt_3_5, 0.0], [0.0, 0.0], [sqrt_3_5, 0.0], [-sqrt_3_5, sqrt_3_5], [0.0, sqrt_3_5], [sqrt_3_5, sqrt_3_5]])
        weights = np.array([25 / 81, 40 / 81, 25 / 81, 40 / 81, 64 / 81, 40 / 81, 25 / 81, 40 / 81, 25 / 81])
    weights *= 4.0 / np.sum(weights)
    return (points, weights)