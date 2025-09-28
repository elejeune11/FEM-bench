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
    F = Fx2
    P_L = F / L
    a_yz = 6.0 / 5.0
    b_yz = 1.0 / 10.0
    c_yz = 2.0 * L / 15.0
    torsional_factor = F * I_rho / (A * L)
    K_geo[1, 1] = a_yz * P_L
    K_geo[5, 5] = c_yz * L * P_L
    K_geo[1, 5] = b_yz * F
    K_geo[5, 1] = b_yz * F
    K_geo[7, 7] = a_yz * P_L
    K_geo[11, 11] = c_yz * L * P_L
    K_geo[7, 11] = -b_yz * F
    K_geo[11, 7] = -b_yz * F
    K_geo[1, 7] = -a_yz * P_L
    K_geo[1, 11] = b_yz * F
    K_geo[5, 7] = -b_yz * F
    K_geo[5, 11] = -L * P_L / 30.0
    K_geo[7, 1] = -a_yz * P_L
    K_geo[11, 1] = b_yz * F
    K_geo[7, 5] = -b_yz * F
    K_geo[11, 5] = -L * P_L / 30.0
    K_geo[2, 2] = a_yz * P_L
    K_geo[4, 4] = c_yz * L * P_L
    K_geo[2, 4] = -b_yz * F
    K_geo[4, 2] = -b_yz * F
    K_geo[8, 8] = a_yz * P_L
    K_geo[10, 10] = c_yz * L * P_L
    K_geo[8, 10] = b_yz * F
    K_geo[10, 8] = b_yz * F
    K_geo[2, 8] = -a_yz * P_L
    K_geo[2, 10] = b_yz * F
    K_geo[4, 8] = b_yz * F
    K_geo[4, 10] = -L * P_L / 30.0
    K_geo[8, 2] = -a_yz * P_L
    K_geo[10, 2] = b_yz * F
    K_geo[8, 4] = b_yz * F
    K_geo[10, 4] = -L * P_L / 30.0
    K_geo[3, 3] = torsional_factor
    K_geo[9, 9] = torsional_factor
    K_geo[3, 9] = -torsional_factor
    K_geo[9, 3] = -torsional_factor
    M_factor = 1.0 / L
    K_geo[3, 4] += 0.5 * Mx2 * M_factor
    K_geo[4, 3] += 0.5 * Mx2 * M_factor
    K_geo[3, 5] += 0.5 * Mx2 * M_factor
    K_geo[5, 3] += 0.5 * Mx2 * M_factor
    K_geo[3, 10] += 0.5 * Mx2 * M_factor
    K_geo[10, 3] += 0.5 * Mx2 * M_factor
    K_geo[3, 11] += 0.5 * Mx2 * M_factor
    K_geo[11, 3] += 0.5 * Mx2 * M_factor
    K_geo[9, 4] += 0.5 * Mx2 * M_factor
    K_geo[4, 9] += 0.5 * Mx2 * M_factor
    K_geo[9, 5] += 0.5 * Mx2 * M_factor
    K_geo[5, 9] += 0.5 * Mx2 * M_factor
    K_geo[9, 10] += 0.5 * Mx2 * M_factor
    K_geo[10, 9] += 0.5 * Mx2 * M_factor
    K_geo[9, 11] += 0.5 * Mx2 * M_factor
    K_geo[11, 9] += 0.5 * Mx2 * M_factor
    K_geo[1, 4] += 0.5 * My1 * M_factor
    K_geo[4, 1] += 0.5 * My1 * M_factor
    K_geo[7, 4] += 0.5 * My1 * M_factor
    K_geo[4, 7] += 0.5 * My1 * M_factor
    K_geo[2, 5] += 0.5 * Mz1 * M_factor
    K_geo[5, 2] += 0.5 * Mz1 * M_factor
    K_geo[8, 5] += 0.5 * Mz1 * M_factor
    K_geo[5, 8] += 0.5 * Mz1 * M_factor
    K_geo[1, 10] += 0.5 * My2 * M_factor
    K