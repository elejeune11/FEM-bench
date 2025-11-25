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
    import numpy as np
    if num_pts == 1:
        x = np.array([0.0], dtype=np.float64)
        w = np.array([2.0], dtype=np.float64)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        x = np.array([-a, a], dtype=np.float64)
        w = np.array([1.0, 1.0], dtype=np.float64)
    elif num_pts == 9:
        a = np.sqrt(3.0 / 5.0)
        x = np.array([-a, 0.0, a], dtype=np.float64)
        w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0], dtype=np.float64)
    else:
        raise ValueError('num_pts must be one of {1, 4, 9}.')
    (xi, eta) = np.meshgrid(x, x, indexing='xy')
    points = np.column_stack((xi.ravel(), eta.ravel()))
    weights = np.outer(w, w).ravel().astype(np.float64, copy=False)
    return (points.astype(np.float64, copy=False), weights)