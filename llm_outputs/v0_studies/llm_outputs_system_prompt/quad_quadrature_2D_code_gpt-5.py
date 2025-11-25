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
        x1d = np.array([0.0], dtype=float)
        w1d = np.array([2.0], dtype=float)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        x1d = np.array([-a, a], dtype=float)
        w1d = np.array([1.0, 1.0], dtype=float)
    elif num_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        x1d = np.array([-a, 0.0, a], dtype=float)
        w1d = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=float)
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    (X, Y) = np.meshgrid(x1d, x1d, indexing='ij')
    points = np.column_stack((X.ravel(), Y.ravel())).astype(float)
    W = np.multiply.outer(w1d, w1d)
    weights = W.ravel().astype(float)
    return (points, weights)