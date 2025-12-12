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
    tol = 1e-08
    v = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float)
    L = np.linalg.norm(v)
    if L < tol:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    e1 = v / L
    if reference_vector is None:
        ez = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(abs(np.dot(e1, ez)) - 1.0) < tol:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            ref = ez
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        ref_norm = np.linalg.norm(ref_arr)
        if abs(ref_norm - 1.0) > tol:
            raise ValueError('reference_vector must be a unit vector.')
        ref = ref_arr
    cross_vec = np.cross(ref, e1)
    cross_norm = np.linalg.norm(cross_vec)
    if cross_norm < tol:
        raise ValueError('reference_vector is parallel to the beam axis.')
    e2 = cross_vec / cross_norm
    e3 = np.cross(e1, e2)
    direction_cosine = np.vstack((e1, e2, e3))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = direction_cosine
    return Gamma