def MSA_3D_transformation_matrix_CC0_H0_T0(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    (dx, dy, dz) = (x2 - x1, y2 - y1, z2 - z1)
    length = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if length < 1e-12:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    ex = np.array([dx / length, dy / length, dz / length])
    if reference_vector is None:
        if abs(ex[2]) > 0.999:
            ref = np.array([0.0, 1.0, 0.0])
        else:
            ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = reference_vector
        if ref.shape != (3,):
            raise ValueError("The reference_vector doesn't have shape (3,).")
        if abs(np.linalg.norm(ref) - 1.0) > 1e-12:
            raise ValueError('reference_vector is not a unit vector.')
        if abs(np.dot(ex, ref)) > 0.999:
            raise ValueError('reference_vector is parallel to the beam axis.')
    ey = np.cross(ref, ex)
    ey_norm = np.linalg.norm(ey)
    if ey_norm < 1e-12:
        raise ValueError('reference_vector is parallel to the beam axis.')
    ey = ey / ey_norm
    ez = np.cross(ex, ey)
    R = np.array([[ex[0], ex[1], ex[2]], [ey[0], ey[1], ey[2]], [ez[0], ez[1], ez[2]]])
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R
    return Gamma