def FEM_2D_quad_quadrature_CC0_H0_T0(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_pts not in {1, 4, 9}:
        raise ValueError('num_pts must be one of {1, 4, 9}')
    if num_pts == 1:
        points = np.array([[0.0, 0.0]], dtype=np.float64)
        weights = np.array([4.0], dtype=np.float64)
    elif num_pts == 4:
        sqrt3_inv = 1.0 / np.sqrt(3.0)
        points = np.array([[-sqrt3_inv, -sqrt3_inv], [sqrt3_inv, -sqrt3_inv], [-sqrt3_inv, sqrt3_inv], [sqrt3_inv, sqrt3_inv]], dtype=np.float64)
        weights = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    elif num_pts == 9:
        sqrt3_inv = 1.0 / np.sqrt(3.0)
        sqrt15_inv = np.sqrt(3.0 / 5.0)
        points = np.array([[-sqrt15_inv, -sqrt15_inv], [0.0, -sqrt15_inv], [sqrt15_inv, -sqrt15_inv], [-sqrt15_inv, 0.0], [0.0, 0.0], [sqrt15_inv, 0.0], [-sqrt15_inv, sqrt15_inv], [0.0, sqrt15_inv], [sqrt15_inv, sqrt15_inv]], dtype=np.float64)
        weights = np.array([25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 64.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0, 40.0 / 81.0, 25.0 / 81.0], dtype=np.float64)
    return (points, weights)