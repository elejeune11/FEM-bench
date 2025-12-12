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
    if L < 1e-14:
        raise ValueError('If the beam has zero length (start and end nodes coincide).')
    local_x = np.array([dx, dy, dz]) / L
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError("If the `reference_vector` doesn't have shape (3,).")
        ref_norm = np.linalg.norm(reference_vector)
        if not np.isclose(ref_norm, 1.0, rtol=1e-06, atol=1e-09):
            raise ValueError('If `reference_vector` is not a unit vector.')
        cross_check = np.cross(reference_vector, local_x)
        if np.linalg.norm(cross_check) < 1e-10:
            raise ValueError('If `reference_vector` is parallel to the beam axis.')
        ref_vec = reference_vector
    else:
        global_z = np.array([0.0, 0.0, 1.0])
        global_y = np.array([0.0, 1.0, 0.0])
        cross_with_z = np.cross(global_z, local_x)
        if np.linalg.norm(cross_with_z) < 1e-10:
            ref_vec = global_y
        else:
            ref_vec = global_z
    local_y = np.cross(ref_vec, local_x)
    local_y = local_y / np.linalg.norm(local_y)
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)
    R = np.array([local_x, local_y, local_z])
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = R
    Gamma[3:6, 3:6] = R
    Gamma[6:9, 6:9] = R
    Gamma[9:12, 9:12] = R
    return Gamma