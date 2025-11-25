def gauss_quadrature_1D(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss points and weights for 1D quadrature over [-1, 1].
    Parameters:
        n (int): Number of Gauss points (1, 2, or 3 recommended)
    Returns:
        tuple[np.ndarray, np.ndarray]: (points, weights)
    """
    if n == 1:
        points = np.array([0.0])
        weights = np.array([2.0])
    elif n == 2:
        sqrt_3_inv = 1.0 / np.sqrt(3)
        points = np.array([-sqrt_3_inv, sqrt_3_inv])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        points = np.array([-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
        weights = np.array([5 / 9, 8 / 9, 5 / 9])
    else:
        raise ValueError('Only 1 to 3 Gauss points are supported.')
    return (points, weights)