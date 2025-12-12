def MSA_3D_local_geometric_stiffness_CC1_H0_T0(L: float, A: float, I_rho: float, Fx2: float, Mx2: float, My1: float, Mz1: float, My2: float, Mz2: float) -> np.ndarray:
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
    if L > 0:
        c1 = Fx2 / (30.0 * L)
        c2 = Fx2 * L / 30.0
        Kg[1, 1] += 36.0 * c1
        Kg[1, 2] += 0.0
        Kg[1, 5] += 3.0 * c2
        Kg[1, 8] -= 36.0 * c1
        Kg[1, 11] += 3.0 * c2
        Kg[2, 2] += 36.0 * c1
        Kg[2, 4] -= 3.0 * c2
        Kg[2, 8] += 0.0
        Kg[2, 10] -= 3.0 * c2
        Kg[4, 4] += 4.0 * L * c1
        Kg[4, 8] -= 3.0 * c2
        Kg[4, 10] -= L * c1
        Kg[5, 5] += 4.0 * L * c1
        Kg[5, 8] -= 3.0 * c2
        Kg[5, 11] -= L * c1
        Kg[8, 8] += 36.0 * c1
        Kg[8, 11] -= 3.0 * c2
        Kg[10, 10] += 4.0 * L * c1
        Kg[11, 11] += 4.0 * L * c1
    if L > 0:
        c_tor = Mx2 / (30.0 * L)
        c_tor_L = Mx2 * L / 30.0
        Kg[2, 3] += 6.0 * c_tor
        Kg[2, 9] -= 6.0 * c_tor
        Kg[3, 5] += 2.0 * L * c_tor
        Kg[3, 8] -= 6.0 * c_tor
        Kg[3, 11] -= L * c_tor
        Kg[5, 9] -= L * c_tor
        Kg[8, 9] += 6.0 * c_tor
        Kg[9, 11] += 2.0 * L * c_tor
        Kg[1, 3] -= 6.0 * c_tor
        Kg[1, 9] += 6.0 * c_tor
        Kg[3, 4] -= 2.0 * L * c_tor
        Kg[3, 7] += 6.0 * c_tor
        Kg[3, 10] += L * c_tor
        Kg[4, 9] += L * c_tor
        Kg[7, 9] -= 6.0 * c_tor
        Kg[9, 10] -= 2.0 * L * c_tor
    if L > 0:
        c_my = (My1 + My2) / (30.0 * L)
        c_my_diff = (My2 - My1) / (30.0 * L)
        Kg[1, 1] += 6.0 * c_my
        Kg[1, 5] += L * c_my_diff
        Kg[1, 7] -= 6.0 * c_my
        Kg[1, 11] += L * c_my_diff
        Kg[5, 5] += 2.0 * L * c_my
        Kg[5, 7] -= L * c_my_diff
        Kg[5, 11] += L * c_my
        Kg[7, 7] += 6.0 * c_my
        Kg[7, 11] -= L * c_my_diff
        Kg[11, 11] += 2.0 * L * c_my
        c_mz = (Mz1 + Mz2) / (30.0 * L)
        c_mz_diff = (Mz2 - Mz1) / (30.0 * L)
        Kg[2, 2] += 6.0 * c_mz
        Kg[2, 4] -= L * c_mz_diff
        Kg[2, 8] -= 6.0 * c_mz
        Kg[2, 10] -= L * c_mz_diff
        Kg[4, 4] += 2.0 * L * c_mz
        Kg[4, 8] += L * c_mz_diff
        Kg[4, 10] -= L * c_mz
        Kg[8, 8] += 6.0 * c_mz
        Kg[8, 10] += L * c_mz_diff
        Kg[10, 10] += 2.0 * L * c_mz
    for i in range(12):
        for j in range(i + 1, 12):
            Kg[j, i] = Kg[i, j]
    return Kg