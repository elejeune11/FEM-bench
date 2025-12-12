def MSA_3D_transformation_matrix_CC0_H0_T0(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    beam_axis = np.array([dx, dy, dz])
    length = np.linalg.norm(beam_axis)
    if length == 0:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    x_local = beam_axis / length
    if reference_vector is None:
        if np.isclose(x_local[0], 0) and np.isclose(x_local[1], 0):
            reference_vector = np.array([0, 1, 0])
        else:
            reference_vector = np.array([0, 0, 1])
    else:
        if reference_vector.shape != (3,):
            raise ValueError("reference_vector doesn't have shape (3,).")
        if not np.isclose(np.linalg.norm(reference_vector), 1):
            raise ValueError('reference_vector is not a unit vector.')
        if np.isclose(np.abs(np.dot(x_local, reference_vector)), 1):
            raise ValueError('reference_vector is parallel to the beam axis.')
    y_local = np.cross(reference_vector, x_local)
    y_local /= np.linalg.norm(y_local)
    z_local = np.cross(x_local, y_local)
    z_local /= np.linalg.norm(z_local)
    T = np.array([x_local, y_local, z_local]).T
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = T
    return Gamma