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
    if abs(Fx2) > 1e-12:
        Kg[1, 1] += Fx2 / L
        Kg[1, 7] -= Fx2 / L
        Kg[7, 1] -= Fx2 / L
        Kg[7, 7] += Fx2 / L
        Kg[2, 2] += Fx2 / L
        Kg[2, 8] -= Fx2 / L
        Kg[8, 2] -= Fx2 / L
        Kg[8, 8] += Fx2 / L
        Kg[4, 4] += Fx2 * L / 15
        Kg[4, 10] -= Fx2 * L / 30
        Kg[10, 4] -= Fx2 * L / 30
        Kg[10, 10] += Fx2 * L / 15
        Kg[5, 5] += Fx2 * L / 15
        Kg[5, 11] -= Fx2 * L / 30
        Kg[11, 5] -= Fx2 * L / 30
        Kg[11, 11] += Fx2 * L / 15
        Kg[1, 5] += Fx2 / 10
        Kg[1, 11] += Fx2 / 10
        Kg[5, 1] += Fx2 / 10
        Kg[11, 1] += Fx2 / 10
        Kg[7, 5] -= Fx2 / 10
        Kg[7, 11] -= Fx2 / 10
        Kg[5, 7] -= Fx2 / 10
        Kg[11, 7] -= Fx2 / 10
        Kg[2, 4] -= Fx2 / 10
        Kg[2, 10] -= Fx2 / 10
        Kg[4, 2] -= Fx2 / 10
        Kg[10, 2] -= Fx2 / 10
        Kg[8, 4] += Fx2 / 10
        Kg[8, 10] += Fx2 / 10
        Kg[4, 8] += Fx2 / 10
        Kg[10, 8] += Fx2 / 10
    if abs(Mx2) > 1e-12:
        Kg[4, 4] += Mx2 / (3 * L)
        Kg[4, 10] += Mx2 / (6 * L)
        Kg[10, 4] += Mx2 / (6 * L)
        Kg[10, 10] += Mx2 / (3 * L)
        Kg[5, 5] += Mx2 / (3 * L)
        Kg[5, 11] += Mx2 / (6 * L)
        Kg[11, 5] += Mx2 / (6 * L)
        Kg[11, 11] += Mx2 / (3 * L)
    My_avg = (My1 + My2) / 2
    if abs(My_avg) > 1e-12:
        Kg[2, 2] += My_avg / L
        Kg[2, 8] -= My_avg / L
        Kg[8, 2] -= My_avg / L
        Kg[8, 8] += My_avg / L
        Kg[4, 4] += My_avg * L / 15
        Kg[4, 10] -= My_avg * L / 30
        Kg[10, 4] -= My_avg * L / 30
        Kg[10, 10] += My_avg * L / 15
    Mz_avg = (Mz1 + Mz2) / 2
    if abs(Mz_avg) > 1e-12:
        Kg[1, 1] += Mz_avg / L
        Kg[1, 7] -= Mz_avg / L
        Kg[7, 1] -= Mz_avg / L
        Kg[7, 7] += Mz_avg / L
        Kg[5, 5] += Mz_avg * L / 15
        Kg[5, 11] -= Mz_avg * L / 30
        Kg[11, 5] -= Mz_avg * L / 30
        Kg[11, 11] += Mz_avg * L / 15
    return Kg