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
    import numpy as np
    K_g = np.zeros((12, 12))
    if abs(Fx2) > 1e-12:
        K_g[1, 1] = Fx2 / L
        K_g[1, 7] = -Fx2 / L
        K_g[7, 1] = -Fx2 / L
        K_g[7, 7] = Fx2 / L
        K_g[2, 2] = Fx2 / L
        K_g[2, 8] = -Fx2 / L
        K_g[8, 2] = -Fx2 / L
        K_g[8, 8] = Fx2 / L
        K_g[4, 4] = Fx2 * L / 15.0
        K_g[4, 10] = -Fx2 * L / 30.0
        K_g[10, 4] = -Fx2 * L / 30.0
        K_g[10, 10] = Fx2 * L / 15.0
        K_g[5, 5] = Fx2 * L / 15.0
        K_g[5, 11] = -Fx2 * L / 30.0
        K_g[11, 5] = -Fx2 * L / 30.0
        K_g[11, 11] = Fx2 * L / 15.0
        K_g[1, 5] = Fx2 / 2.0
        K_g[5, 1] = Fx2 / 2.0
        K_g[1, 11] = Fx2 / 2.0
        K_g[11, 1] = Fx2 / 2.0
        K_g[7, 5] = -Fx2 / 2.0
        K_g[5, 7] = -Fx2 / 2.0
        K_g[7, 11] = -Fx2 / 2.0
        K_g[11, 7] = -Fx2 / 2.0
        K_g[2, 4] = -Fx2 / 2.0
        K_g[4, 2] = -Fx2 / 2.0
        K_g[2, 10] = -Fx2 / 2.0
        K_g[10, 2] = -Fx2 / 2.0
        K_g[8, 4] = Fx2 / 2.0
        K_g[4, 8] = Fx2 / 2.0
        K_g[8, 10] = Fx2 / 2.0
        K_g[10, 8] = Fx2 / 2.0
    if abs(Mx2) > 1e-12:
        K_g[3, 3] = Mx2 / (3.0 * I_rho)
        K_g[3, 9] = -Mx2 / (6.0 * I_rho)
        K_g[9, 3] = -Mx2 / (6.0 * I_rho)
        K_g[9, 9] = Mx2 / (3.0 * I_rho)
        K_g[3, 4] = Mx2 / (2.0 * I_rho)
        K_g[4, 3] = Mx2 / (2.0 * I_rho)
        K_g[3, 10] = Mx2 / (2.0 * I_rho)
        K_g[10, 3] = Mx2 / (2.0 * I_rho)
        K_g[9, 4] = -Mx2 / (2.0 * I_rho)
        K_g[4, 9] = -Mx2 / (2.0 * I_rho)
        K_g[9, 10] = -Mx2 / (2.0 * I_rho)
        K_g[10, 9] = -Mx2 / (2.0 * I_rho)
        K_g[3, 5] = -Mx2 / (2.0 * I_rho)
        K_g[5, 3] = -Mx2 / (2.0 * I_rho)
        K_g[3, 11] = -Mx2 / (2.0 * I_rho)
        K_g[11, 3] = -Mx2 / (2.0 * I_rho)
        K_g[9, 5] = Mx2 / (2.0 * I_rho)
        K_g[5, 9] = Mx2 / (2.0 * I_rho)
        K_g[9, 11] = Mx2 / (2.0 * I_rho)
        K_g[11, 9] = Mx2 / (2.0 * I_rho)
    if abs(My1) > 1e-12:
        K_g[2, 4] += My1 / L
        K_g[4, 2] += My1 / L
        K_g[8, 4] -= My1 / L
        K_g[4, 8] -= My1 / L
        K_g[4, 4] += My1 / (3.0 * L)
        K_g[4, 10] += My1 / (6.0 * L)
        K_g[10, 4] += My1 / (6.0 * L)
        K_g[10, 10] -= My1 / (3.0 * L)
    if abs(My2) > 1e-12:
        K_g[2, 10] += My2 / L
        K_g[10, 2] += My2 / L
        K_g[8, 10] -= My2 / L
        K_g[10, 8] -= My2 / L
        K_g[4, 4] -= My2 / (3.0 * L)
        K_g[4, 10] -= My2 / (6.0 * L)
        K_g[10, 4] -= My2 / (6.0 * L)
        K_g[10, 10] += My2 / (3.0 * L)
    if abs(Mz1) > 1e-12:
        K_g[1, 5] -= Mz1 / L
        K_g[5, 1] -= Mz1 / L
        K_g[7, 5] += Mz1 / L
        K_g[5, 7] += Mz1 / L
        K_g[5, 5] += Mz1 / (3.0 * L)
        K_g[5, 11] += Mz1 / (6.0 * L)
        K_g[11, 5] += Mz1 / (6.0 * L)
        K_g[11, 11] -= Mz1 / (3.0 * L)
    if abs(Mz2) > 1e-12:
        K_g[1, 11] -= Mz2 / L
        K_g[11, 1] -= Mz2 / L
        K_g[7, 11] += Mz2 / L
        K_g[11, 7] += Mz2 / L
        K_g[5, 5] -= Mz2 / (3.0 * L)
        K_g[5, 11] -= Mz2 / (6.0 * L)
        K_g[11, 5] -= Mz2 / (6.0 * L)
        K_g[11, 11] += Mz2 / (3.0 * L)
    return K_g