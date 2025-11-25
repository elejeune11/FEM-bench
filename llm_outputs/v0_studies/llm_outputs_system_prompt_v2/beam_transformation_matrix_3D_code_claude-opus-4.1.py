def beam_transformation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, reference_vector: Optional[np.ndarray]) -> np.ndarray:
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
    beam_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
    beam_length = np.linalg.norm(beam_vector)
    if beam_length < 1e-10:
        raise ValueError('Beam has zero length (start and end nodes coincide).')
    local_x = beam_vector / beam_length
    if reference_vector is not None:
        if reference_vector.shape != (3,):
            raise ValueError("reference_vector doesn't have shape (3,).")
        ref_norm = np.linalg.norm(reference_vector)
        if abs(ref_norm - 1.0) > 1e-10:
            raise ValueError('reference_vector is not a unit vector.')
        cross_product = np.cross(reference_vector, local_x)
        cross_norm = np.linalg.norm(cross_product)
        if cross_norm < 1e-10:
            raise ValueError('reference_vector is parallel to the beam axis.')
        local_y = cross_product / cross_norm
    else:
        if abs(local_x[2]) > 0.999:
            reference_vector = np.array([0.0, 1.0, 0.0])
        else:
            reference_vector = np.array([0.0, 0.0, 1.0])
        cross_product = np.cross(reference_vector, local_x)
        local_y = cross_product / np.linalg.norm(cross_product)
    local_z = np.cross(local_x, local_y)
    T = np.zeros((3, 3))
    T[0, :] = local_x
    T[1, :] = local_y
    T[2, :] = local_z
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = T
    Gamma[3:6, 3:6] = T
    Gamma[6:9, 6:9] = T
    Gamma[9:12, 9:12] = T
    return Gamma