def beam_transformation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    if np.allclose([x1, y1, z1], [x2, y2, z2]):
        raise ValueError('Beam has zero length (start and end nodes coincide).')
    x_axis = np.array([x2 - x1, y2 - y1, z2 - z1])
    x_axis = x_axis / np.linalg.norm(x_axis)
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        if not np.isclose(np.linalg.norm(reference_vector), 1.0):
            raise ValueError('reference_vector must be a unit vector.')
        cross_prod = np.cross(reference_vector, x_axis)
        if np.allclose(cross_prod, 0.0):
            raise ValueError('reference_vector is parallel to the beam axis.')
        y_axis = np.cross(reference_vector, x_axis)
    else:
        if np.allclose(x_axis, [0, 0, 1]) or np.allclose(x_axis, [0, 0, -1]):
            y_axis = np.array([0, 1, 0])
        else:
            y_axis = np.array([0, 0, 1])
        cross_prod = np.cross(y_axis, x_axis)
        if np.allclose(cross_prod, 0.0):
            y_axis = np.array([1, 0, 0])
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return Gamma