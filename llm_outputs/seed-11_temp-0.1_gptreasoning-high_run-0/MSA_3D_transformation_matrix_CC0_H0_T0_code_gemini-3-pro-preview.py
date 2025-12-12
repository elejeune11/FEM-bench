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
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    if np.isclose(L, 0.0):
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    vx = np.array([dx, dy, dz]) / L
    if reference_vector is None:
        if np.isclose(np.abs(vx[2]), 1.0):
            ref_vec = np.array([0.0, 1.0, 0.0])
        else:
            ref_vec = np.array([0.0, 0.0, 1.0])
    else:
        if reference_vector.shape != (3,):
            raise ValueError("reference_vector doesn't have shape (3,).")
        if not np.isclose(np.linalg.norm(reference_vector), 1.0):
            raise ValueError('reference_vector is not a unit vector.')
        if np.isclose(np.linalg.norm(np.cross(reference_vector, vx)), 0.0):
            raise ValueError('reference_vector is parallel to the beam axis.')
        ref_vec = reference_vector
    vy_raw = np.cross(ref_vec, vx)
    vy = vy_raw / np.linalg.norm(vy_raw)
    vz = np.cross(vx, vy)
    Lambda = np.array([vx, vy, vz])
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = Lambda
    Gamma[3:6, 3:6] = Lambda
    Gamma[6:9, 6:9] = Lambda
    Gamma[9:12, 9:12] = Lambda
    return Gamma