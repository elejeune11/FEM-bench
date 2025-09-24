def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 10.0
    A = 1.0
    I_rho = 0.5
    K_g_zero = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert K_g_zero.shape == (12, 12), 'Matrix shape should be 12x12'
    assert np.allclose(K_g_zero, np.zeros((12, 12))), 'Matrix should be zero for zero loads'
    assert np.allclose(K_g_zero, K_g_zero.T), 'Zero matrix must be symmetric'
    Fx_tension = 100.0
    K_g_tension = fcn(L, A, I_rho, Fx_tension, 0, 0, 0, 0, 0)
    assert K_g_tension.shape == (12, 12), 'Matrix shape should be 12x12'
    assert np.allclose(K_g_tension, K_g_tension.T), 'Matrix should be symmetric'
    assert K_g_tension[1, 1] > 0
    assert K_g_tension[2, 2] > 0
    assert K_g_tension[7, 7] > 0
    assert K_g_tension[8, 8] > 0
    Fx_compression = -100.0
    K_g_compression = fcn(L, A, I_rho, Fx_compression, 0, 0, 0, 0, 0)
    assert np.allclose(K_g_compression, K_g_compression.T), 'Matrix should be symmetric'
    assert K_g_compression[1, 1] < 0
    assert K_g_compression[2, 2] < 0
    assert K_g_compression[7, 7] < 0
    assert K_g_compression[8, 8] < 0
    K_g_2Fx = fcn(L, A, I_rho, 2 * Fx_tension, 0, 0, 0, 0, 0)
    assert np.allclose(K_g_2Fx, 2 * K_g_tension), 'Matrix should scale linearly with Fx2'
    Mx2 = 50.0
    K_g_torsion = fcn(L, A, I_rho, 0, Mx2, 0, 0, 0, 0)
    assert not np.allclose(K_g_torsion, np.zeros((12, 12))), 'Matrix should be non-zero for torsional load'
    assert np.allclose(K_g_torsion, K_g_torsion.T), 'Matrix should be symmetric'
    My1 = 60.0
    Mz2 = 70.0
    K_g_bending = fcn(L, A, I_rho, 0, 0, My1, 0, 0, Mz2)
    assert not np.allclose(K_g_bending, np.zeros((12, 12))), 'Matrix should be non-zero for bending loads'
    assert np.allclose(K_g_bending, K_g_bending.T), 'Matrix should be symmetric'

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
Compare numerical result with the analytical Euler buckling load.
Design the test so that comparison tolerances account for discretization error.
    """
    E = 200000000000.0
    L = 5.0
    b = 0.1
    h = 0.1
    A = b * h
    I_y = b * h ** 3 / 12
    I_z = h * b ** 3 / 12
    I_rho = I_y + I_z
    G = E / (2 * (1 + 0.3))
    K_e = np.zeros((12, 12))
    K_e[0, 0] = K_e[6, 6] = E * A / L
    K_e[0, 6] = K_e[6, 0] = -E * A / L
    K_e[3, 3] = K_e[9, 9] = G * I_rho / L
    K_e[3, 9] = K_e[9, 3] = -G * I_rho / L
    c1z = 12 * E * I_z / L ** 3
    c2z = 6 * E * I_z / L ** 2
    c3z = 4 * E * I_z / L
    c4z = 2 * E * I_z / L
    (K_e[1, 1], K_e[7, 7]) = (c1z, c1z)
    (K_e[1, 7], K_e[7, 1]) = (-c1z, -c1z)
    (K_e[1, 5], K_e[5, 1]) = (c2z, c2z)
    (K_e[1, 11], K_e[11, 1]) = (c2z, c2z)
    (K_e[5, 7], K_e[7, 5]) = (-c2z, -c2z)
    (K_e[7, 11], K_e[11, 7]) = (-c2z, -c2z)
    (K_e[5, 5], K_e[11, 11]) = (c3z, c3z)
    (K_e[5, 11], K_e[11, 5]) = (c4z, c4z)
    c1y = 12 * E * I_y / L ** 3
    c2y = 6 * E * I_y / L ** 2
    c3y = 4 * E * I_y / L
    c4y = 2 * E * I_y / L
    (K_e[2, 2], K_e[8, 8]) = (c1y, c1y)
    (K_e[2, 8], K_e[8, 2]) = (-c1y, -c1y)
    (K_e[2, 4], K_e[4, 2]) = (-c2y, -c2y)
    (K_e[2, 10], K_e[10, 2]) = (-c2y, -c2y)
    (K_e[4, 8], K_e[8, 4]) = (c2y, c2y)
    (K_e[8, 10], K_e[10, 8]) = (c2y, c2y)
    (K_e[4, 4], K_e[10, 10]) = (c3y, c3y)
    (K_e[4, 10], K_e[10, 4]) = (c4y, c4y)
    K_g_unit = fcn(L, A, I_rho, -1.0, 0, 0, 0, 0, 0)
    free_dofs = slice(6, 12)
    K_e_ff = K_e[free_dofs, free_dofs]
    K_g_unit_ff = K_g_unit[free_dofs, free_dofs]
    (eigenvalues, _) = la.eig(K_e_ff, -K_g_unit_ff)
    positive_real_eigenvalues = np.real(eigenvalues[np.isreal(eigenvalues) & (eigenvalues > 1e-06)])
    numerical_p_cr = np.min(positive_real_eigenvalues)
    analytical_p_cr = np.pi ** 2 * E * I_y / (4 * L ** 2)
    assert numerical_p_cr == pytest.approx(analytical_p_cr, rel=1e-06)