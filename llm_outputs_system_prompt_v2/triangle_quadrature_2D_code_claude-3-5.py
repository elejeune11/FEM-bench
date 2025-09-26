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
    if num_pts not in {1, 3, 4}:
        raise ValueError('Number of quadrature points must be 1, 3, or 4')
    if num_pts == 1:
        points = np.array([[1 / 3, 1 / 3]], dtype=np.float64)
        weights = np.array([0.5], dtype=np.float64)
    elif num_pts == 3:
        points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]], dtype=np.float64)
        weights = np.full(3, 1 / 6, dtype=np.float64)
    else:
        alpha = 0.6
        beta = 0.2
        points = np.array([[1 / 3, 1 / 3], [alpha, beta], [beta, alpha], [beta, beta]], dtype=np.float64)
        weights = np.array([-0.28125, 0.260416667, 0.260416667, 0.260416667], dtype=np.float64)
    return (points, weights)