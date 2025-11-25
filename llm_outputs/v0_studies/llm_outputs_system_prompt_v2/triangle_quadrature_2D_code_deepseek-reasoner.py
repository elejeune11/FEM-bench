def triangle_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_pts == 1:
        points = np.array([[1 / 3, 1 / 3]], dtype=np.float64)
        weights = np.array([1 / 2], dtype=np.float64)
    elif num_pts == 3:
        points = np.array([[1 / 6, 1 / 6], [1 / 6, 2 / 3], [2 / 3, 1 / 6]], dtype=np.float64)
        weights = np.array([1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
    elif num_pts == 4:
        points = np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]], dtype=np.float64)
        weights = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96], dtype=np.float64)
    else:
        raise ValueError('num_pts must be 1, 3, or 4')
    return (points, weights)