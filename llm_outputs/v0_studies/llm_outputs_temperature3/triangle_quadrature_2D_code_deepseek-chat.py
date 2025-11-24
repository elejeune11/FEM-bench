def triangle_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return quadrature points and weights for numerical integration
    over the reference triangle T = {(x,y): x>=0, y>=0, x+y<=1}.
    Supported rules:
    Parameters
    ----------
    num_pts : int
        Number of quadrature points (1, 3, or 4).
    Returns
    -------
    points : (num_pts, 2) ndarray of float64
        Quadrature points (xi, eta). The third barycentric coordinate is 1 - xi - eta.
    weights : (num_pts,) ndarray of float64
        Quadrature weights. The sum of weights equals the area of the reference triangle (1/2).
    Raises
    ------
    ValueError
        If `num_pts` is not 1, 3, or 4.
    """
    if num_pts == 1:
        points = np.array([[1 / 3, 1 / 3]], dtype=np.float64)
        weights = np.array([0.5], dtype=np.float64)
    elif num_pts == 3:
        points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=np.float64)
        weights = np.array([1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
    elif num_pts == 4:
        alpha1 = 1 / 3
        alpha2 = 0.6
        alpha3 = 0.2
        beta1 = -9 / 16
        beta2 = 25 / 48
        points = np.array([[alpha1, alpha1], [alpha2, alpha3], [alpha3, alpha2], [alpha3, alpha3]], dtype=np.float64)
        weights = np.array([beta1, beta2, beta2, beta2], dtype=np.float64)
    else:
        raise ValueError('num_pts must be 1, 3, or 4')
    return (points, weights)