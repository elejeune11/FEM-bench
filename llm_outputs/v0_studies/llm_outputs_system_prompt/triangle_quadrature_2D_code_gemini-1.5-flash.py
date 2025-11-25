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
        points = np.array([[1.0 / 3, 1.0 / 3]])
        weights = np.array([1.0 / 2])
    elif num_pts == 3:
        points = np.array([[1.0 / 6, 1.0 / 6], [1.0 / 6, 2.0 / 3], [2.0 / 3, 1.0 / 6]])
        weights = np.array([1.0 / 6, 1.0 / 6, 1.0 / 6]) * 2
    elif num_pts == 4:
        points = np.array([[1.0 / 3, 1.0 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
        weights = np.array([1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6]) * 2
    else:
        raise ValueError('Unsupported number of quadrature points.')
    return (points, weights)