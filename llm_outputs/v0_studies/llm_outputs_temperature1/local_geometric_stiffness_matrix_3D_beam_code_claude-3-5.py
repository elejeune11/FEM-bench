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
    if Fx2 != 0:
        Kg[1, 1] = Kg[7, 7] = 6 / 5 * Fx2 / L
        Kg[1, 7] = Kg[7, 1] = -6 / 5 * Fx2 / L
        Kg[1, 5] = Kg[5, 1] = Kg[7, 11] = Kg[11, 7] = Fx2 / 10
        Kg[1, 11] = Kg[11, 1] = Kg[5, 7] = Kg[7, 5] = -Fx2 / 10
        Kg[5, 5] = Kg[11, 11] = 2 * L / 15 * Fx2
        Kg[5, 11] = Kg[11, 5] = -L / 30 * Fx2
        Kg[2, 2] = Kg[8, 8] = 6 / 5 * Fx2 / L
        Kg[2, 8] = Kg[8, 2] = -6 / 5 * Fx2 / L
        Kg[2, 4] = Kg[4, 2] = Kg[8, 10] = Kg[10, 8] = -Fx2 / 10
        Kg[2, 10] = Kg[10, 2] = Kg[4, 8] = Kg[8, 4] = Fx2 / 10
        Kg[4, 4] = Kg[10, 10] = 2 * L / 15 * Fx2
        Kg[4, 10] = Kg[10, 4] = -L / 30 * Fx2
    if Mx2 != 0:
        Kg[1, 4] = Kg[4, 1] = -Mx2 / (2 * L)
        Kg[1, 10] = Kg[10, 1] = -Mx2 / (2 * L)
        Kg[7, 4] = Kg[4, 7] = -Mx2 / (2 * L)
        Kg[7, 10] = Kg[10, 7] = -Mx2 / (2 * L)
        Kg[2, 5] = Kg[5, 2] = Mx2 / (2 * L)
        Kg[2, 11] = Kg[11, 2] = Mx2 / (2 * L)
        Kg[8, 5] = Kg[5, 8] = Mx2 / (2 * L)
        Kg[8, 11] = Kg[11, 8] = Mx2 / (2 * L)
    if My1 != 0 or My2 != 0:
        Kg[2, 3] = Kg[3, 2] = -My1 / (2 * L)
        Kg[2, 9] = Kg[9, 2] = -My2 / (2 * L)
        Kg[8, 3] = Kg[3, 8] = -My1 / (2 * L)
        Kg[8, 9] = Kg[9, 8] = -My2 / (2 * L)
    if Mz1 != 0 or Mz2 != 0:
        Kg[1, 3] = Kg[3, 1] = Mz1 / (2 * L)
        Kg[1, 9] = Kg[9, 1] = Mz2 / (2 * L)
        Kg[7, 3] = Kg[3, 7] = Mz1 / (2 * L)
        Kg[7, 9] = Kg[9, 7] = Mz2 / (2 * L)
    return Kg