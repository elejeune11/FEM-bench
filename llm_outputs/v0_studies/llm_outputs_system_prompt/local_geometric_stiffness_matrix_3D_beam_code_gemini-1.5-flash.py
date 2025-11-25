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
    Notes:
    Effects captured
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
    Implementation details
    """
    Kg = np.zeros((12, 12))
    Kg[0, 6] = -Fx2
    Kg[6, 0] = -Fx2
    Kg[1, 7] = -Fx2 / 2
    Kg[7, 1] = -Fx2 / 2
    Kg[2, 8] = -Fx2 / 2
    Kg[8, 2] = -Fx2 / 2
    Kg[1, 1] = Fx2 * L / 3
    Kg[2, 2] = Fx2 * L / 3
    Kg[7, 7] = Fx2 * L / 3
    Kg[8, 8] = Fx2 * L / 3
    Kg[1, 7] = -Fx2 * L / 6
    Kg[7, 1] = -Fx2 * L / 6
    Kg[2, 8] = -Fx2 * L / 6
    Kg[8, 2] = -Fx2 * L / 6
    Kg[3, 9] = -Mx2
    Kg[9, 3] = -Mx2
    Kg[3, 3] = Mx2 * L / 3
    Kg[9, 9] = Mx2 * L / 3
    Kg[3, 9] = -Mx2 * L / 6
    Kg[9, 3] = -Mx2 * L / 6
    Kg[4, 10] = My1
    Kg[10, 4] = My1
    Kg[4, 4] = -My1 * L / 3
    Kg[10, 10] = -My1 * L / 3
    Kg[4, 10] = My1 * L / 6
    Kg[10, 4] = My1 * L / 6
    Kg[5, 11] = Mz1
    Kg[11, 5] = Mz1
    Kg[5, 5] = -Mz1 * L / 3
    Kg[11, 11] = -Mz1 * L / 3
    Kg[5, 11] = Mz1 * L / 6
    Kg[11, 5] = Mz1 * L / 6
    Kg[4, 10] = My2
    Kg[10, 4] = My2
    Kg[4, 4] = -My2 * L / 3
    Kg[10, 10] = -My2 * L / 3
    Kg[4, 10] = My2 * L / 6
    Kg[10, 4] = My2 * L / 6
    Kg[5, 11] = Mz2
    Kg[11, 5] = Mz2
    Kg[5, 5] = -Mz2 * L / 3
    Kg[11, 11] = -Mz2 * L / 3
    Kg[5, 11] = Mz2 * L / 6
    Kg[11, 5] = Mz2 * L / 6
    return Kg