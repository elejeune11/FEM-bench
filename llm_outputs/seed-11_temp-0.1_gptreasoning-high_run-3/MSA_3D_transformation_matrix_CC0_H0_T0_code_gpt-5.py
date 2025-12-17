def MSA_3D_transformation_matrix_CC0_H0_T0(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
    """
    Compute the 12x12 transformation matrix Gamma for a 3D beam element.
    This transformation relates the element's local coordinate system to the global system:
        K_global = Gamma.T @ K_local @ Gamma
    where K_global is the global stiffness matrix and K_local is the local stiffness matrix.
    Parameters:
        x1, y1, z1 (float): Coordinates of the beam's start node in global space.
        x2, y2, z2 (float): Coordinates of the beam's end node in global space.
        reference_vector (np.ndarray of shape (3,), optional): A unit vector in global coordinates used to define
            the orientation of the local y-axis. The local y-axis is computed as the cross product
            of the reference vector and the local x-axis (beam axis). The local z-axis is then
            computed as the cross product of the local x-axis and the local y-axes.
            If not provided:
    Returns:
        Gamma (np.ndarray): A 12x12 local-to-global transformation matrix used to transform
            stiffness matrices, displacements, and forces. It is composed of four repeated
            3x3 direction cosine submatrices along the diagonal.
    Raises:
        ValueError: If `reference_vector` is not a unit vector.
        ValueError: If `reference_vector` is parallel to the beam axis.
        ValueError: If the `reference_vector` doesn't have shape (3,).
        ValueError: If the beam has zero length (start and end nodes coincide).
    Notes:
        All vectors must be specified in a right-handed global Cartesian coordinate system.
    """
    p1 = np.array([x1, y1, z1], dtype=float)
    p2 = np.array([x2, y2, z2], dtype=float)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L == 0.0:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    ex = d / L
    tol = 1e-08
    if reference_vector is None:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(np.dot(ex, z_axis)) >= 1.0 - tol:
            ref = y_axis
        else:
            ref = z_axis
        ref_unit = ref
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.shape != (3,):
            raise ValueError('`reference_vector` must have shape (3,).')
        if not np.all(np.isfinite(ref_arr)):
            raise ValueError('`reference_vector` is not a unit vector.')
        ref_norm = np.linalg.norm(ref_arr)
        if not np.isfinite(ref_norm) or abs(ref_norm - 1.0) > tol:
            raise ValueError('`reference_vector` is not a unit vector.')
        ref_unit = ref_arr / ref_norm
        if abs(np.dot(ref_unit, ex)) >= 1.0 - tol:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
    ey_temp = np.cross(ref_unit, ex)
    ey_norm = np.linalg.norm(ey_temp)
    if ey_norm == 0.0:
        raise ValueError('`reference_vector` is parallel to the beam axis.')
    ey = ey_temp / ey_norm
    ez_temp = np.cross(ex, ey)
    ez_norm = np.linalg.norm(ez_temp)
    if ez_norm == 0.0:
        raise ValueError('Failed to compute a valid local coordinate system.')
    ez = ez_temp / ez_norm
    R = np.column_stack((ex, ey, ez))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        idx = slice(3 * i, 3 * (i + 1))
        Gamma[idx, idx] = R
    return Gamma