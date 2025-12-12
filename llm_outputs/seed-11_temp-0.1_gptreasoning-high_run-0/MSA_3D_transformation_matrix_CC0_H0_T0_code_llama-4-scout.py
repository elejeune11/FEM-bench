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
    beam_axis = np.array([x2 - x1, y2 - y1, z2 - z1])
    beam_length = np.linalg.norm(beam_axis)
    if beam_length == 0:
        raise ValueError('Beam has zero length (start and end nodes coincide).')
    beam_axis = beam_axis / beam_length
    if reference_vector is None:
        if np.isclose(beam_axis[0], 0) and np.isclose(beam_axis[1], 0):
            reference_vector = np.array([0, 1, 0])
        else:
            reference_vector = np.array([0, 0, 1])
    if reference_vector.shape != (3,):
        raise ValueError('Reference vector must have shape (3,).')
    if not np.isclose(np.linalg.norm(reference_vector), 1):
        raise ValueError('Reference vector is not a unit vector.')
    if np.isclose(np.dot(beam_axis, reference_vector), 1):
        raise ValueError('Reference vector is parallel to the beam axis.')
    local_y_axis = np.cross(reference_vector, beam_axis)
    local_y_axis = local_y_axis / np.linalg.norm(local_y_axis)
    local_z_axis = np.cross(beam_axis, local_y_axis)
    local_axes = np.column_stack((beam_axis, local_y_axis, local_z_axis))
    transformation_matrix = np.eye(12)
    for i in range(4):
        transformation_matrix[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = local_axes
    return transformation_matrix