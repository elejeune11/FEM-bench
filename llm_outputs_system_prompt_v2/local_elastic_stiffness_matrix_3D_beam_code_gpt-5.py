def local_elastic_stiffness_matrix_3D_beam(E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
    """
    Return the 12×12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.
    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:
        [force_vector] = [stiffness_matrix] @ [displacement_vector]
    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Where:
    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio (used for torsion only)
        A (float): Cross-sectional area
        L (float): Length of the beam element
        Iy (float): Second moment of area about the local y-axis
        Iz (float): Second moment of area about the local z-axis
        J (float): Torsional constant
    Returns:
        np.ndarray: A 12×12 symmetric stiffness matrix representing axial, torsional,
                    and bending stiffness in local coordinates.
    """
    import numpy as np
    E = float(E)
    nu = float(nu)
    A = float(A)
    L = float(L)
    Iy = float(Iy)
    Iz = float(Iz)
    J = float(J)
    if L <= 0.0:
        raise ValueError('Beam length L must be positive.')
    K = np.zeros((12, 12), dtype=float)
    a = E * A / L if A != 0.0 and E != 0.0 else 0.0
    if J != 0.0 and E != 0.0:
        denom = 2.0 * (1.0 + nu)
        if denom == 0.0:
            raise ValueError("Invalid Poisson's ratio: 1 + nu must not be zero for nonzero torsional constant J.")
        t = E / denom * J / L
    else:
        t = 0.0
    facz = E * Iz / L ** 3 if Iz != 0.0 and E != 0.0 else 0.0
    b1 = 12.0 * facz
    b2 = 6.0 * L * facz
    b3 = 4.0 * L * L * facz
    b4 = 2.0 * L * L * facz
    facy = E * Iy / L ** 3 if Iy != 0.0 and E != 0.0 else 0.0
    c1 = 12.0 * facy
    c2 = 6.0 * L * facy
    c3 = 4.0 * L * L * facy
    c4 = 2.0 * L * L * facy
    (i_u, i_v, i_w, i_tx, i_ty, i_tz) = (0, 1, 2, 3, 4, 5)
    (j_u, j_v, j_w, j_tx, j_ty, j_tz) = (6, 7, 8, 9, 10, 11)
    if a != 0.0:
        K[i_u, i_u] += a
        K[i_u, j_u] += -a
        K[j_u, i_u] += -a
        K[j_u, j_u] += a
    if t != 0.0:
        K[i_tx, i_tx] += t
        K[i_tx, j_tx] += -t
        K[j_tx, i_tx] += -t
        K[j_tx, j_tx] += t
    if facz != 0.0:
        K[i_v, i_v] += b1
        K[i_v, i_tz] += b2
        K[i_v, j_v] += -b1
        K[i_v, j_tz] += b2
        K[i_tz, i_v] += b2
        K[i_tz, i_tz] += b3
        K[i_tz, j_v] += -b2
        K[i_tz, j_tz] += b4
        K[j_v, i_v] += -b1
        K[j_v, i_tz] += -b2
        K[j_v, j_v] += b1
        K[j_v, j_tz] += -b2
        K[j_tz, i_v] += b2
        K[j_tz, i_tz] += b4
        K[j_tz, j_v] += -b2
        K[j_tz, j_tz] += b3
    if facy != 0.0:
        K[i_w, i_w] += c1
        K[i_w, i_ty] += -c2
        K[i_w, j_w] += -c1
        K[i_w, j_ty] += -c2
        K[i_ty, i_w] += -c2
        K[i_ty, i_ty] += c3
        K[i_ty, j_w] += c2
        K[i_ty, j_ty] += c4
        K[j_w, i_w] += -c1
        K[j_w, i_ty] += c2
        K[j_w, j_w] += c1
        K[j_w, j_ty] += c2
        K[j_ty, i_w] += -c2
        K[j_ty, i_ty] += c4
        K[j_ty, j_w] += c2
        K[j_ty, j_ty] += c3
    return K