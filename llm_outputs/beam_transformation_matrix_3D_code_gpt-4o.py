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
    beam_axis = np.array([x2 - x1, y2 - y1, z2 - z1])
    beam_length = np.linalg.norm(beam_axis)
    if beam_length == 0:
        raise ValueError('The beam has zero length (start and end nodes coincide).')
    local_x = beam_axis / beam_length
    if reference_vector is None:
        if np.allclose(local_x, [0, 0, 1]):
            reference_vector = np.array([0, 1, 0])
        else:
            reference_vector = np.array([0, 0, 1])
    else:
        if reference_vector.shape != (3,):
            raise ValueError("The `reference_vector` doesn't have shape (3,).")
        if not np.isclose(np.linalg.norm(reference_vector), 1):
            raise ValueError('`reference_vector` is not a unit vector.')
    local_y = np.cross(reference_vector, local_x)
    local_y_norm = np.linalg.norm(local_y)
    if local_y_norm == 0:
        raise ValueError('`reference_vector` is parallel to the beam axis.')
    local_y /= local_y_norm
    local_z = np.cross(local_x, local_y)
    direction_cosines = np.array([local_x, local_y, local_z])
    Gamma = np.zeros((12, 12))
    for i in range(4):
        Gamma[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = direction_cosines
    return Gamma