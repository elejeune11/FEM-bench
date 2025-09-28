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
        nodes = np.array([0.0], dtype=np.float64)
        weights_1d = np.array([2.0], dtype=np.float64)
    elif num_pts == 4:
        r3 = np.sqrt(3.0)
        nodes = np.array([-1.0 / r3, 1.0 / r3], dtype=np.float64)
        weights_1d = np.array([1.0, 1.0], dtype=np.float64)
    elif num_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        nodes = np.array([-a, 0.0, a], dtype=np.float64)
        weights_1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}.')
    (X, Y) = np.meshgrid(nodes, nodes, indexing='xy')
    points = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float64)
    weights = np.outer(weights_1d, weights_1d).ravel().astype(np.float64)
    return (points, weights)