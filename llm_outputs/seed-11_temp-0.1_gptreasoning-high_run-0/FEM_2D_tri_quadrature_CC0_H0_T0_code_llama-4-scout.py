def FEM_2D_tri_quadrature_CC0_H0_T0(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return quadrature points and weights for numerical integration
    over the reference triangle T = {(x, y): x >= 0, y >= 0, x + y <= 1}.
    Supported rules (standard FEM triangular quadrature)
    ----------------------------------------------------
    1-point rule (degree-1 exact):
        Points and weights:
            (1/3, 1/3)      → weight = 0.5
    3-point rule (degree-2 exact):
        Points are the symmetric permutations of (1/6, 1/6),
        returned in the following fixed order:
            (1/6, 1/6)      → weight = 1/6
            (2/3, 1/6)      → weight = 1/6
            (1/6, 2/3)      → weight = 1/6
    4-point rule (degree-3 exact):
        Points include the centroid plus symmetric permutations of (0.6, 0.2),
        returned in the following fixed order:
            (1/3, 1/3)      → weight = -27/96
            (0.6, 0.2)      → weight =  25/96
            (0.2, 0.6)      → weight =  25/96
            (0.2, 0.2)      → weight =  25/96
    Parameters
    ----------
    num_pts : int
        Number of quadrature points. Must be one of {1, 3, 4}.
    Returns
    -------
    points : (num_pts, 2) ndarray of float64
        Quadrature points (ξ, η). The third barycentric coordinate is 1 − ξ − η.
        Points are returned in the fixed canonical order listed above.
    weights : (num_pts,) ndarray of float64
        Corresponding quadrature weights. The sum of weights equals 0.5,
        the area of the reference triangle.
    Raises
    ------
    ValueError
        If `num_pts` is not one of {1, 3, 4}.
    Notes
    -----
      triangular elements.
    """
    if num_pts == 1:
        points = np.array([[1 / 3, 1 / 3]])
        weights = np.array([0.5])
    elif num_pts == 3:
        points = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
        weights = np.array([1 / 6, 1 / 6, 1 / 6])
    elif num_pts == 4:
        points = np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]])
        weights = np.array([-27 / 96, 25 / 96, 25 / 96, 25 / 96])
    else:
        raise ValueError('num_pts must be one of {1, 3, 4}')
    return (points, weights)