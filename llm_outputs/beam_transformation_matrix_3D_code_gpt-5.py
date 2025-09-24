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
    import numpy as np
    from typing import Optional
    import pytest
    tol = 1e-12
    p1 = np.array([x1, y1, z1], dtype=float)
    p2 = np.array([x2, y2, z2], dtype=float)
    axis = p2 - p1
    L = np.linalg.norm(axis)
    if L <= tol:
        raise ValueError('Beam has zero length (start and end nodes coincide).')
    x_hat = axis / L
    if reference_vector is None:
        z_global = np.array([0.0, 0.0, 1.0], dtype=float)
        y_global = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(np.dot(x_hat, z_global)) >= 1.0 - 1e-08:
            ref = y_global
        else:
            ref = z_global
    else:
        ref_arr = np.asarray(reference_vector, dtype=float)
        if ref_arr.shape != (3,):
            raise ValueError('reference_vector must have shape (3,).')
        ref_norm = np.linalg.norm(ref_arr)
        if not np.isfinite(ref_norm) or abs(ref_norm - 1.0) > 1e-08:
            raise ValueError('reference_vector must be a unit vector.')
        ref = ref_arr
    cross_ref_x = np.cross(ref, x_hat)
    n_cross = np.linalg.norm(cross_ref_x)
    if n_cross <= 1e-12:
        raise ValueError('reference_vector is parallel (or nearly parallel) to the beam axis.')
    y_hat = cross_ref_x / n_cross
    z_hat = np.cross(x_hat, y_hat)
    z_norm = np.linalg.norm(z_hat)
    if z_norm <= tol:
        raise ValueError('Failed to construct a valid local coordinate system.')
    z_hat = z_hat / z_norm
    y_hat = np.cross(z_hat, x_hat)
    y_norm = np.linalg.norm(y_hat)
    if y_norm <= tol:
        raise ValueError('Failed to construct a valid local coordinate system.')
    y_hat = y_hat / y_norm
    R = np.column_stack((x_hat, y_hat, z_hat))
    Gamma = np.zeros((12, 12), dtype=float)
    for i in range(4):
        Gamma[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    return Gamma