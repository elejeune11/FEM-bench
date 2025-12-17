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
    r1 = np.array([x1, y1, z1], dtype=float)
    r2 = np.array([x2, y2, z2], dtype=float)
    v = r2 - r1
    L = np.linalg.norm(v)
    if L <= 1e-12:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    ex = v / L
    if reference_vector is None:
        gz = np.array([0.0, 0.0, 1.0])
        gy = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(ex, gz)) >= 1.0 - 1e-08:
            ref = gy
        else:
            ref = gz
    else:
        ref = np.asarray(reference_vector, dtype=float)
        if ref.shape != (3,):
            raise ValueError('`reference_vector` must have shape (3,).')
        nref = np.linalg.norm(ref)
        if not np.isclose(nref, 1.0, atol=1e-08):
            raise ValueError('`reference_vector` must be a unit vector.')
        ref = ref / nref
        if abs(np.dot(ref, ex)) >= 1.0 - 1e-08:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
    ey = np.cross(ref, ex)
    ney = np.linalg.norm(ey)
    if ney <= 1e-12:
        if reference_vector is not None:
            raise ValueError('`reference_vector` is parallel to the beam axis.')
        gx = np.array([1.0, 0.0, 0.0])
        gy = np.array([0.0, 1.0, 0.0])
        cand = gy if abs(np.dot(ex, gy)) < 1.0 - 1e-08 else gx
        ey = np.cross(cand, ex)
        ney = np.linalg.norm(ey)
        if ney <= 1e-12:
            raise ValueError('Failed to compute local axes due to numerical issues.')
    ey = ey / ney
    ez = np.cross(ex, ey)
    nez = np.linalg.norm(ez)
    if nez <= 1e-12:
        raise ValueError('Failed to compute local axes due to numerical issues.')
    ez = ez / nez
    R = np.column_stack((ex, ey, ez))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        Gamma[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = R
    return Gamma