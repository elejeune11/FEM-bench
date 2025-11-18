def beam_transformation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    length = np.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        raise ValueError('Beam has zero length (start and end nodes coincide)')
    x_axis = np.array([dx, dy, dz]) / length
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError('reference_vector must have shape (3,)')
        ref_norm = np.linalg.norm(reference_vector)
        if abs(ref_norm - 1.0) > 1e-12:
            raise ValueError('reference_vector must be a unit vector')
        y_axis = np.cross(reference_vector, x_axis)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-12:
            raise ValueError('reference_vector is parallel to the beam axis')
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)
    else:
        if abs(x_axis[0]) < 1e-12 and abs(x_axis[1]) < 1e-12:
            ref_vector = np.array([0.0, 1.0, 0.0])
        else:
            ref_vector = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(ref_vector, x_axis)
        y_norm = np.linalg.norm(y_axis)
        y_axis /= y_norm
        z_axis = np.cross(x_axis, y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
    return Gamma