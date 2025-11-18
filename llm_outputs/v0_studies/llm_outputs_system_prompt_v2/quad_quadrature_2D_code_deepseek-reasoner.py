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
        points_1d = np.array([0.0])
        weights_1d = np.array([2.0])
    elif num_pts == 4:
        points_1d = np.array([-0.5773502691896257, 0.5773502691896257])
        weights_1d = np.array([1.0, 1.0])
    elif num_pts == 9:
        points_1d = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
        weights_1d = np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    (xi, eta) = np.meshgrid(points_1d, points_1d, indexing='ij')
    points = np.column_stack((xi.ravel(), eta.ravel()))
    (w_xi, w_eta) = np.meshgrid(weights_1d, weights_1d, indexing='ij')
    weights = (w_xi * w_eta).ravel()
    return (points, weights)