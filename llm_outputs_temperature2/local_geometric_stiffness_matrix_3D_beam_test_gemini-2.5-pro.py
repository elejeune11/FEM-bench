def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 10.0
    A = 0.1
    I_rho = 0.01
    K_g_zero = fcn(L=L, A=A, I_rho=I_rho, Fx2=0, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    assert K_g_zero.shape == (12, 12), 'Matrix shape should be 12x12'
    assert np.all(K_g_zero == 0), 'Matrix should be all zeros for zero loads'
    K_g_full = fcn(L=L, A=A, I_rho=I_rho, Fx2=1000.0, Mx2=100, My1=50, Mz1=-50, My2=-50, Mz2=50)
    assert K_g_full.shape == (12, 12), 'Matrix shape should be 12x12'
    assert np.allclose(K_g_full, K_g_full.T), 'Matrix should be symmetric'
    Fx2_tension = 1200.0
    K_g_tension = fcn(L=L, A=A, I_rho=I_rho, Fx2=Fx2_tension, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    assert K_g_tension[1, 1] == pytest.approx(6 / 5 * Fx2_tension / L)
    assert K_g_tension[2, 2] == pytest.approx(6 / 5 * Fx2_tension / L)
    assert K_g_tension[1, 1] > 0
    assert K_g_tension[4, 4] == pytest.approx(2 / 15 * Fx2_tension * L)
    assert K_g_tension[5, 5] == pytest.approx(2 / 15 * Fx2_tension * L)
    assert K_g_tension[4, 4] > 0
    Fx2_compression = -1200.0
    K_g_compression = fcn(L=L, A=A, I_rho=I_rho, Fx2=Fx2_compression, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    assert K_g_compression[1, 1] == pytest.approx(6 / 5 * Fx2_compression / L)
    assert K_g_compression[1, 1] < 0
    assert K_g_compression[4, 4] == pytest.approx(2 / 15 * Fx2_compression * L)
    assert K_g_compression[4, 4] < 0
    K_g_base = fcn(L=L, A=A, I_rho=I_rho, Fx2=1.0, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    K_g_scaled = fcn(L=L, A=A, I_rho=I_rho, Fx2=500.0, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    assert np.allclose(K_g_scaled, 500.0 * K_g_base)
    Mx2_val = 20.0
    K_g_torsion = fcn(L=L, A=A, I_rho=I_rho, Fx2=0, Mx2=Mx2_val, My1=0, Mz1=0, My2=0, Mz2=0)
    assert not np.all(K_g_torsion == 0)
    assert K_g_torsion[4, 8] == pytest.approx(Mx2_val / 2.0)
    assert K_g_torsion[5, 7] == pytest.approx(-Mx2_val / 2.0)
    My1_val = 30.0
    K_g_bend_y = fcn(L=L, A=A, I_rho=I_rho, Fx2=0, Mx2=0, My1=My1_val, Mz1=0, My2=0, Mz2=0)
    assert not np.all(K_g_bend_y == 0)
    assert K_g_bend_y[2, 6] == pytest.approx(-My1_val / L)
    assert K_g_bend_y[6, 2] == pytest.approx(-My1_val / L)
    Mz2_val = 40.0
    K_g_bend_z = fcn(L=L, A=A, I_rho=I_rho, Fx2=0, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=Mz2_val)
    assert not np.all(K_g_bend_z == 0)
    assert K_g_bend_z[0, 7] == pytest.approx(Mz2_val / L)
    assert K_g_bend_z[7, 0] == pytest.approx(Mz2_val / L)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """

    def _get_elastic_stiffness_matrix(E, G, A, L, Iy, Iz, J):
        """Helper to create the 12x12 elastic stiffness matrix for a 3D beam."""
        Ke = np.zeros((12, 12))
        k_ax = E * A / L
        Ke[0, 0] = Ke[6, 6] = k_ax
        Ke[0, 6] = Ke[6, 0] = -k_ax
        k_tor = G * J / L
        Ke[3, 3] = Ke[9, 9] = k_tor
        Ke[3, 9] = Ke[9, 3] = -k_tor
        k11_y = 12 * E * Iy / L ** 3
        k12_y = 6 * E * Iy / L ** 2
        k22_y = 4 * E * Iy / L
        k24_y = 2 * E * Iy / L
        dofs_y = np.ix_([2, 4, 8, 10], [2, 4, 8, 10])
        Ke[dofs_y] = np.array([[k11_y, k12_y, -k11_y, k12_y], [k12_y, k22_y, -k12_y, k24_y], [-k11_y, -k12_y, k11_y, -k12_y], [k12_y, k24_y, -k12_y, k22_y]])
        k11_z = 12 * E * Iz / L ** 3
        k12_z = 6 * E * Iz / L ** 2
        k22_z = 4 * E * Iz / L
        k24_z = 2 * E * Iz / L
        dofs_z = np.ix_([1, 5, 7, 11], [1, 5, 7, 11])
        Ke[dofs_z] = np.array([[k11_z, -k12_z, -k11_z, -k12_z], [-k12_z, k22_z, k12_z, k24_z], [-k11_z, k12_z, k11_z, k12_z], [-k12_z, k24_z, k12_z, k22_z]])
        return Ke
    L = 10.0
    E = 210000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    A = 0.001
    Iy = 1e-06
    Iz = 2e-06
    J = Iy + Iz
    I_rho = A * L ** 2 / 12
    Ke = _get_elastic_stiffness_matrix(E, G, A, L, Iy, Iz, J)
    Kg_unit = fcn(L=L, A=A, I_rho=I_rho, Fx2=-1.0, Mx2=0, My1=0, Mz1=0, My2=0, Mz2=0)
    free_dofs = np.arange(6, 12)
    Ke_free = Ke[np.ix_(free_dofs, free_dofs)]
    Kg_free = Kg_unit[np.ix_(free_dofs, free_dofs)]
    (eigenvalues, _) = eig(Ke_free, -Kg_free)
    positive_real_eigenvalues = np.real(eigenvalues[np.isreal(eigenvalues) & (np.real(eigenvalues) > 1e-06)])
    if len(positive_real_eigenvalues) == 0:
        pytest.fail('No positive real eigenvalues found for buckling load.')
    numerical_pcr = np.min(positive_real_eigenvalues)
    X = (6 - np.sqrt(36 - 4 * 0.15 * 12)) / (2 * 0.15)
    expected_fem_pcr = X * (E * Iy) / L ** 2
    assert numerical_pcr == pytest.approx(expected_fem_pcr, rel=1e-09)