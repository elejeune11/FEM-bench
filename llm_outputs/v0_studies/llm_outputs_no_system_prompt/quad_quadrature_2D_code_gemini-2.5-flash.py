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
        n_1d = 1
    elif num_pts == 4:
        n_1d = 2
    elif num_pts == 9:
        n_1d = 3
    else:
        raise ValueError(f'Unsupported number of quadrature points: {num_pts}. Must be 1, 4, or 9.')
    (xi_1d, w_1d) = np.polynomial.legendre.leggauss(n_1d)
    (xi_grid, eta_grid) = np.meshgrid(xi_1d, xi_1d)
    points = np.vstack((xi_grid.ravel(), eta_grid.ravel())).T
    weights = np.outer(w_1d, w_1d).ravel()
    return (points, weights)