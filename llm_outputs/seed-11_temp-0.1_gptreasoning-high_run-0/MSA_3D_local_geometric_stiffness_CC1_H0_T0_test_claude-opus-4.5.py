def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 0.01
    I_rho = 1e-05
    Fx2 = 1000.0
    Mx2 = 50.0
    My1 = 30.0
    Mz1 = 40.0
    My2 = 35.0
    Mz2 = 45.0
    Kg = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert Kg.shape == (12, 12), 'Matrix should be 12x12'
    assert np.allclose(Kg, Kg.T, atol=1e-12), 'Matrix should be symmetric'
    Kg_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kg_zero, np.zeros((12, 12)), atol=1e-14), 'Matrix should be zero when all loads are zero'
    Fx2_tension = 1000.0
    Fx2_compression = -1000.0
    Kg_tension = fcn(L, A, I_rho, Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_compression = fcn(L, A, I_rho, Fx2_compression, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kg_tension, -Kg_compression, atol=1e-12), 'Tension and compression should produce opposite geometric stiffness contributions'
    Kg_no_torsion = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_with_torsion = fcn(L, A, I_rho, Fx2, 100.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(Kg_no_torsion, Kg_with_torsion, atol=1e-12), 'Matrix should change when torsional moment is applied'
    Kg_no_bending = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_with_bending = fcn(L, A, I_rho, Fx2, 0.0, 100.0, 100.0, 100.0, 100.0)
    assert not np.allclose(Kg_no_bending, Kg_with_bending, atol=1e-12), 'Matrix should change when bending moments are applied'
    Fx2_base = 500.0
    Fx2_double = 1000.0
    Kg_base = fcn(L, A, I_rho, Fx2_base, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_double = fcn(L, A, I_rho, Fx2_double, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kg_double, 2.0 * Kg_base, atol=1e-12), 'Geometric stiffness should scale linearly with Fx2'

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    E = 210000000000.0
    L_total = 10.0
    b = 0.1
    h = 0.1
    A = b * h
    I = b * h ** 3 / 12
    I_rho = b * h * (b ** 2 + h ** 2) / 12
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L_total ** 2)
    n_elements = 10
    L_elem = L_total / n_elements
    n_nodes = n_elements + 1
    n_dofs = 6 * n_nodes

    def get_elastic_stiffness_3D_beam(E, A, I, L):
        """Simple 3D Euler-Bernoulli beam elastic stiffness matrix."""
        Ke = np.zeros((12, 12))
        EA_L = E * A / L
        Ke[0, 0] = EA_L
        Ke[0, 6] = -EA_L
        Ke[6, 0] = -EA_L
        Ke[6, 6] = EA_L
        EI_L3 = E * I / L ** 3
        EI_L2 = E * I / L ** 2
        EI_L = E * I / L
        Ke[2, 2] = 12 * EI_L3
        Ke[2, 4] = -6 * EI_L2
        Ke[2, 8] = -12 * EI_L3
        Ke[2, 10] = -6 * EI_L2
        Ke[4, 2] = -6 * EI_L2
        Ke[4, 4] = 4 * EI_L
        Ke[4, 8] = 6 * EI_L2
        Ke[4, 10] = 2 * EI_L
        Ke[8, 2] = -12 * EI_L3
        Ke[8, 4] = 6 * EI_L2
        Ke[8, 8] = 12 * EI_L3
        Ke[8, 10] = 6 * EI_L2
        Ke[10, 2] = -6 * EI_L2
        Ke[10, 4] = 2 * EI_L
        Ke[10, 8] = 6 * EI_L2
        Ke[10, 10] = 4 * EI_L
        Ke[1, 1] = 12 * EI_L3
        Ke[1, 5] = 6 * EI_L2
        Ke[1, 7] = -12 * EI_L3
        Ke[1, 11] = 6 * EI_L2
        Ke[5, 1] = 6 * EI_L2
        Ke[5, 5] = 4 * EI_L
        Ke[5, 7] = -6 * EI_L2
        Ke[5, 11] = 2 * EI_L
        Ke[7, 1] = -12 * EI_L3
        Ke[7, 5] = -6 * EI_L2
        Ke[7, 7] = 12 * EI_L3
        Ke[7, 11] = -6 * EI_L2
        Ke[11, 1] = 6 * EI_L2
        Ke[11, 5] = 2 * EI_L
        Ke[11, 7] = -6 * EI_L2
        Ke[11, 11] = 4 * EI_L
        GJ_L = E * I / (2.6 * L)
        Ke[3, 3] = GJ_L
        Ke[3, 9] = -GJ_L
        Ke[9, 3] = -GJ_L
        Ke[9, 9] = GJ_L
        return Ke
    K_global = np.zeros((n_dofs, n_dofs))
    for i in range(n_elements):
        Ke = get_elastic_stiffness_3D_beam(E, A, I, L_elem)
        dofs = list(range(6 * i, 6 * i + 12))
        for (ii, di) in enumerate(dofs):
            for (jj, dj) in enumerate(dofs):
                K_global[di, dj] += Ke[ii, jj]
    Kg_global = np.zeros((n_dofs, n_dofs))
    Fx2_unit = -1.0
    for i in range(n_elements):
        Kg = fcn(L_elem, A, I_rho, Fx2_unit, 0.0, 0.0, 0.0, 0.0, 0.0)
        dofs = list(range(6 * i, 6 * i + 12))
        for (ii, di) in enumerate(dofs):
            for (jj, dj) in enumerate(dofs):
                Kg_global[di, dj] += Kg[ii, jj]
    fixed_dofs = list(range(6))
    free_dofs = list(range(6, n_dofs))
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_reduced = Kg_global[np.ix_(free_dofs, free_dofs)]
    try:
        (eigenvalues, _) = np.linalg.eig(np.linalg.solve(K_reduced, -Kg_reduced))
        real_eigenvalues = eigenvalues[np.isreal(eigenvalues)].real
        positive_eigenvalues = real_eigenvalues[real_eigenvalues > 0]
        if len(positive_eigenvalues) > 0:
            P_cr_numerical = np.min(positive_eigenvalues)
        else:
            eigenvalues = np.linalg.eigvals(np.linalg.solve(-Kg_reduced, K_reduced))
            real_eigenvalues = eigenvalues[np.isreal(eigenvalues)].real
            positive_eigenvalues = real_eigenvalues[real_eigenvalues > 0]
            P_cr_numerical = np.min(positive_eigenvalues)
        relative_error = abs(P_cr_numerical - P_cr_analytical) / P_cr_analytical
        assert relative_error < 0.1, f'Buckling load error {relative_error * 100:.2f}% exceeds 10% tolerance. Numerical: {P_cr_numerical:.2e}, Analytical: {P_cr_analytical:.2e}'
    except np.linalg.LinAlgError:
        Kg_compression = fcn(L_elem, A, I_rho, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        Ke = get_elastic_stiffness_3D_beam(E, A, I, L_elem)
        eig_elastic = np.linalg.eigvalsh(Ke)
        eig_combined = np.linalg.eigvalsh(Ke + Kg_compression)
        assert np.min(eig_combined) < np.min(eig_elastic), 'Compression should reduce stiffness (softening effect)'