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
    K_geo = np.zeros((12, 12))
    if L != 0:
        Fx_term = Fx2 / L
        K_geo[1, 1] = Fx_term
        K_geo[5, 5] = Fx_term * L ** 2 / 6
        K_geo[1, 5] = -Fx_term * L / 2
        K_geo[5, 1] = -Fx_term * L / 2
        K_geo[2, 2] = Fx_term
        K_geo[4, 4] = Fx_term * L ** 2 / 6
        K_geo[2, 4] = Fx_term * L / 2
        K_geo[4, 2] = Fx_term * L / 2
        K_geo[7, 7] = Fx_term
        K_geo[11, 11] = Fx_term * L ** 2 / 6
        K_geo[7, 11] = Fx_term * L / 2
        K_geo[11, 7] = Fx_term * L / 2
        K_geo[8, 8] = Fx_term
        K_geo[10, 10] = Fx_term * L ** 2 / 6
        K_geo[8, 10] = -Fx_term * L / 2
        K_geo[10, 8] = -Fx_term * L / 2
        K_geo[1, 7] = -Fx_term
        K_geo[7, 1] = -Fx_term
        K_geo[1, 11] = Fx_term * L / 2
        K_geo[11, 1] = Fx_term * L / 2
        K_geo[5, 7] = Fx_term * L / 2
        K_geo[7, 5] = Fx_term * L / 2
        K_geo[5, 11] = -Fx_term * L ** 2 / 3
        K_geo[11, 5] = -Fx_term * L ** 2 / 3
        K_geo[2, 8] = -Fx_term
        K_geo[8, 2] = -Fx_term
        K_geo[2, 10] = -Fx_term * L / 2
        K_geo[10, 2] = -Fx_term * L / 2
        K_geo[4, 8] = -Fx_term * L / 2
        K_geo[8, 4] = -Fx_term * L / 2
        K_geo[4, 10] = -Fx_term * L ** 2 / 3
        K_geo[10, 4] = -Fx_term * L ** 2 / 3
    if L != 0:
        torsion_term = Mx2 / L
        K_geo[1, 4] = torsion_term
        K_geo[4, 1] = torsion_term
        K_geo[2, 5] = -torsion_term
        K_geo[5, 2] = -torsion_term
        K_geo[1, 10] = -torsion_term
        K_geo[10, 1] = -torsion_term
        K_geo[2, 11] = torsion_term
        K_geo[11, 2] = torsion_term
        K_geo[4, 7] = -torsion_term
        K_geo[7, 4] = -torsion_term
        K_geo[5, 8] = torsion_term
        K_geo[8, 5] = torsion_term
        K_geo[4, 10] = torsion_term
        K_geo[10, 4] = torsion_term
        K_geo[5, 11] = -torsion_term
        K_geo[11, 5] = -torsion_term
        K_geo[7, 10] = torsion_term
        K_geo[10, 7] = torsion_term
        K_geo[8, 11] = -torsion_term
        K_geo[11, 8] = -torsion_term
    if L != 0:
        K_geo[2, 3] = My1 / L
        K_geo[3, 2] = My1 / L
        K_geo[2, 9] = -My1 / L
        K_geo[9, 2] = -My1 / L
        K_geo[3, 8] = -My1 / L
        K_geo[8, 3] = -My1 / L
        K_geo[8, 9] = My1 / L
        K_geo[9, 8] = My1 / L
        K_geo[1, 3] = -Mz1 / L
        K_geo[3, 1] = -Mz1 / L
        K_geo[1, 9] = Mz1 / L
        K_geo[9, 1] = Mz1 / L
        K_geo[3, 7] = Mz1 / L
        K_geo[7, 3] = Mz1 / L
        K_geo[7, 9] = -Mz1 / L
        K_geo[9, 7] = -Mz1 / L
        K_geo[2, 3] += My2 / L
        K_geo[3, 2] += My2 / L
        K_geo[2, 9] += My2 / L
        K_geo[9, 2] += My2 / L
        K_geo[3, 8] += My2 / L
        K_geo[8, 3] += My2 / L
        K_geo[8, 9] += -My2 / L
        K_geo[9, 8] += -My2 / L
        K_geo[1, 3] += Mz2 / L
        K_geo[3, 1] += Mz2 / L
        K_geo[1, 9] += -Mz2 / L
        K_geo[9, 1] += -Mz2 / L
        K_geo[3, 7] += -Mz2 / L
        K_geo[7, 3] += -Mz2 / L
        K_geo[7, 9] += Mz2 / L
        K_geo[9, 7] += Mz2 / L
    return K_geo