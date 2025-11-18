def local_geometric_stiffness_matrix_3D_beam(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
    """
    Return the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.
    The beam is assumed to be aligned with the local x-axis. The geometric stiffness matrix is used in conjunction with the elastic stiffness matrix for nonlinear structural analysis.
    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]
    Where:
    Parameters:
        L (float): Length of the beam element [length units]
        A (float): Cross-sectional area [length² units]
        I_rho (float): Polar moment of inertia about the x-axis [length⁴ units]
        Fx2 (float): Internal axial force in the element (positive = tension), evaluated at node 2 [force units]
        Mx2 (float): Torsional moment at node 2 about x-axis [forcexlength units]
        My1 (float): Bending moment at node 1 about y-axis [forcexlength units]
        Mz1 (float): Bending moment at node 1 about z-axis [forcexlength units]
        My2 (float): Bending moment at node 2 about y-axis [forcexlength units]
        Mz2 (float): Bending moment at node 2 about z-axis [forcexlength units]
    Returns:
        np.ndarray: A 12x12 symmetric geometric stiffness matrix in local coordinates.
                    Positive axial force (tension) contributes to element stiffness;
                    negative axial force (compression) can lead to instability.
    """
    Kg = np.zeros((12, 12))
    if abs(Fx2) > 0:
        Fx = Fx2 / L
        Kg[1, 1] = 6 * Fx / 5
        Kg[1, 5] = Fx * L / 10
        Kg[1, 7] = -6 * Fx / 5
        Kg[1, 11] = Fx * L / 10
        Kg[5, 5] = 2 * Fx * L * L / 15
        Kg[5, 7] = -Fx * L / 10
        Kg[5, 11] = -Fx * L * L / 30
        Kg[7, 7] = 6 * Fx / 5
        Kg[7, 11] = -Fx * L / 10
        Kg[11, 11] = 2 * Fx * L * L / 15
        Kg[2, 2] = 6 * Fx / 5
        Kg[2, 4] = -Fx * L / 10
        Kg[2, 8] = -6 * Fx / 5
        Kg[2, 10] = -Fx * L / 10
        Kg[4, 4] = 2 * Fx * L * L / 15
        Kg[4, 8] = Fx * L / 10
        Kg[4, 10] = -Fx * L * L / 30
        Kg[8, 8] = 6 * Fx / 5
        Kg[8, 10] = Fx * L / 10
        Kg[10, 10] = 2 * Fx * L * L / 15
    if abs(Mx2) > 0:
        Mx = Mx2 / L
        Kg[1, 3] = -Mx / L
        Kg[2, 3] = Mx / L
        Kg[3, 7] = Mx / L
        Kg[3, 8] = -Mx / L
    if abs(My1) > 0 or abs(My2) > 0:
        Kg[2, 3] += (My1 + My2) / (2 * L)
        Kg[3, 8] += -(My1 + My2) / (2 * L)
    if abs(Mz1) > 0 or abs(Mz2) > 0:
        Kg[1, 3] += -(Mz1 + Mz2) / (2 * L)
        Kg[3, 7] += (Mz1 + Mz2) / (2 * L)
    for i in range(12):
        for j in range(i):
            Kg[i, j] = Kg[j, i]
    return Kg