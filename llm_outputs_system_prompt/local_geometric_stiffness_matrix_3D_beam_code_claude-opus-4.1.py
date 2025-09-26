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
    P = Fx2
    c1 = 6.0 / (5.0 * L)
    c2 = 1.0 / 10.0
    c3 = 2.0 * L / 15.0
    c4 = -L / 30.0
    Kg[1, 1] = P * c1
    Kg[1, 5] = P * c2
    Kg[1, 7] = -P * c1
    Kg[1, 11] = P * c2
    Kg[5, 1] = P * c2
    Kg[5, 5] = P * c3
    Kg[5, 7] = -P * c2
    Kg[5, 11] = P * c4
    Kg[7, 1] = -P * c1
    Kg[7, 5] = -P * c2
    Kg[7, 7] = P * c1
    Kg[7, 11] = -P * c2
    Kg[11, 1] = P * c2
    Kg[11, 5] = P * c4
    Kg[11, 7] = -P * c2
    Kg[11, 11] = P * c3
    Kg[2, 2] = P * c1
    Kg[2, 4] = -P * c2
    Kg[2, 8] = -P * c1
    Kg[2, 10] = -P * c2
    Kg[4, 2] = -P * c2
    Kg[4, 4] = P * c3
    Kg[4, 8] = P * c2
    Kg[4, 10] = P * c4
    Kg[8, 2] = -P * c1
    Kg[8, 4] = P * c2
    Kg[8, 8] = P * c1
    Kg[8, 10] = P * c2
    Kg[10, 2] = -P * c2
    Kg[10, 4] = P * c4
    Kg[10, 8] = P * c2
    Kg[10, 10] = P * c3
    T = Mx2
    t_factor = T * I_rho / (A * L)
    Kg[4, 4] += t_factor
    Kg[4, 10] += -t_factor
    Kg[5, 5] += t_factor
    Kg[5, 11] += -t_factor
    Kg[10, 4] += -t_factor
    Kg[10, 10] += t_factor
    Kg[11, 5] += -t_factor
    Kg[11, 11] += t_factor
    my_avg = (My1 + My2) / 2.0
    my_diff = (My2 - My1) / L
    Kg[2, 0] += -my_diff / L
    Kg[2, 6] += my_diff / L
    Kg[0, 2] += -my_diff / L
    Kg[6, 2] += my_diff / L
    Kg[0, 8] += my_diff / L
    Kg[8, 0] += my_diff / L
    Kg[6, 8] += -my_diff / L
    Kg[8, 6] += -my_diff / L
    mz_avg = (Mz1 + Mz2) / 2.0
    mz_diff = (Mz2 - Mz1) / L
    Kg[1, 0] += mz_diff / L
    Kg[1, 6] += -mz_diff / L
    Kg[0, 1] += mz_diff / L
    Kg[6, 1] += -mz_diff / L
    Kg[0, 7] += -mz_diff / L
    Kg[7, 0] += -mz_diff / L
    Kg[6, 7] += mz_diff / L
    Kg[7, 6] += mz_diff / L
    return Kg