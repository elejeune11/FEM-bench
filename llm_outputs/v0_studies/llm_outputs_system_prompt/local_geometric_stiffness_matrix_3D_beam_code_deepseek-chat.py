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
    K_g = np.zeros((12, 12))
    a = Fx2 / L
    K_g[1, 1] = a
    K_g[1, 7] = -a
    K_g[7, 1] = -a
    K_g[7, 7] = a
    K_g[2, 2] = a
    K_g[2, 8] = -a
    K_g[8, 2] = -a
    K_g[8, 8] = a
    b_y = Mx2 / (2 * L)
    b_z = Mx2 / (2 * L)
    K_g[1, 5] = b_y
    K_g[1, 11] = b_y
    K_g[5, 1] = b_y
    K_g[11, 1] = b_y
    K_g[7, 5] = -b_y
    K_g[7, 11] = -b_y
    K_g[5, 7] = -b_y
    K_g[11, 7] = -b_y
    K_g[2, 4] = -b_z
    K_g[2, 10] = -b_z
    K_g[4, 2] = -b_z
    K_g[10, 2] = -b_z
    K_g[8, 4] = b_z
    K_g[8, 10] = b_z
    K_g[4, 8] = b_z
    K_g[10, 8] = b_z
    c_my = 1.0 / L
    K_g[1, 4] = -My1 * c_my
    K_g[4, 1] = -My1 * c_my
    K_g[1, 10] = -My2 * c_my
    K_g[10, 1] = -My2 * c_my
    K_g[7, 4] = My1 * c_my
    K_g[4, 7] = My1 * c_my
    K_g[7, 10] = My2 * c_my
    K_g[10, 7] = My2 * c_my
    c_mz = 1.0 / L
    K_g[2, 5] = Mz1 * c_mz
    K_g[5, 2] = Mz1 * c_mz
    K_g[2, 11] = Mz2 * c_mz
    K_g[11, 2] = Mz2 * c_mz
    K_g[8, 5] = -Mz1 * c_mz
    K_g[5, 8] = -Mz1 * c_mz
    K_g[8, 11] = -Mz2 * c_mz
    K_g[11, 8] = -Mz2 * c_mz
    d_my = 1.0 / 2.0
    d_mz = 1.0 / 2.0
    K_g[4, 4] += My1 * d_my
    K_g[4, 10] += My2 * d_my
    K_g[10, 4] += My2 * d_my
    K_g[10, 10] += My2 * d_my
    K_g[5, 5] += Mz1 * d_mz
    K_g[5, 11] += Mz2 * d_mz
    K_g[11, 5] += Mz2 * d_mz
    K_g[11, 11] += Mz2 * d_mz
    return K_g