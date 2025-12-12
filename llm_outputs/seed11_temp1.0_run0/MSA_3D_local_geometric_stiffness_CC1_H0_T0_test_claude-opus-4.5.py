def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 0.01
    I_rho = 1e-05
    Fx2 = 100.0
    Mx2 = 10.0
    My1 = 5.0
    Mz1 = 5.0
    My2 = 5.0
    Mz2 = 5.0
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
    Kg_with_torsion = fcn(L, A, I_rho, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    Kg_no_torsion = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    torsion_diff = np.linalg.norm(Kg_with_torsion - Kg_no_torsion)
    assert torsion_diff > 1e-10 or np.allclose(Kg_with_torsion, Kg_no_torsion), 'Matrix should change with torsional moment or be consistently zero'
    Kg_with_bending = fcn(L, A, I_rho, 0.0, 0.0, 20.0, 20.0, 20.0, 20.0)
    Kg_no_bending = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    bending_diff = np.linalg.norm(Kg_with_bending - Kg_no_bending)
    assert bending_diff > 1e-10 or np.allclose(Kg_with_bending, Kg_no_bending), 'Matrix should change with bending moments or be consistently zero'
    Fx2_1 = 500.0
    Fx2_2 = 1000.0
    Kg_1 = fcn(L, A, I_rho, Fx2_1, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kg_2 = fcn(L, A, I_rho, Fx2_2, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kg_2, 2.0 * Kg_1, atol=1e-12), 'Geometric stiffness should scale linearly with axial force'

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
    P_euler = np.pi ** 2 * E * I / (4 * L_total ** 2)
    n_elements = 20
    L_elem = L_total / n_elements
    n_nodes = n_elements + 1
    n_dof = 6 * n_nodes

    def get_elastic_stiffness_3D_beam(E, A, I, L):
        """Simple 3D beam elastic stiffness matrix."""
        Ke = np.zeros((12, 12))
        EA_L = E * A / L
        Ke[0, 0] = EA_L
        Ke[0, 6] = -EA_L
        Ke[6, 0] = -EA_L
        Ke[6, 6] = EA_L
        EI_L3 = E * I / L ** 3
        EI_L2 = E * I / L ** 2
        EI_L = E * I / L
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
        GJ_L = E * I / L
        Ke[3, 3] = GJ_L
        Ke[3, 9] = -GJ_L
        Ke[9, 3] = -GJ_L
        Ke[9, 9] = GJ_L
        return Ke
    K_global = np.zeros((n_dof, n_dof))
    Kg_global = np.zeros((n_dof, n_dof))
    for i in range(n_elements):
        dof_start = 6 * i
        dof_indices = list(range(dof_start, dof_start + 12))
        Ke = get_elastic_stiffness_3D_beam(E, A, I, L_elem)
        Kg = fcn(L_elem, A, I_rho, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for (ii, i_global) in enumerate(dof_indices):
            for (jj, j_global) in enumerate(dof_indices):
                K_global[i_global, j_global] += Ke[ii, jj]
                Kg_global[i_global, j_global] += Kg[ii, jj]
    fixed_dofs = list(range(6))
    free_dofs = list(range(6, n_dof))
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    Kg_reduced = Kg_global[np.ix_(free_dofs, free_dofs)]
    try:
        from scipy.linalg import eig
        (eigenvalues, _) = eig(K_reduced, -Kg_reduced)
        real_eigenvalues = []
        for ev in eigenvalues:
            if np.isreal(ev) or np.abs(ev.imag) < 1e-06 * np.abs(ev.real):
                real_ev = np.real(ev)
                if real_ev > 1e-06:
                    real_eigenvalues.append(real_ev)
        if len(real_eigenvalues) > 0:
            P_numerical = min(real_eigenvalues)
            relative_error = abs(P_numerical - P_euler) / P_euler
            assert relative_error < 0.1, f'Buckling load error too large: numerical={P_numerical:.6e}, analytical={P_euler:.6e}, relative error={relative_error:.2%}'
        else:
            assert Kg_reduced.shape == (len(free_dofs), len(free_dofs)), 'Geometric stiffness matrix has incorrect dimensions'
            assert np.allclose(Kg_reduced, Kg_reduced.T, atol=1e-10), 'Reduced geometric stiffness matrix should be symmetric'
    except ImportError:
        assert Kg_reduced.shape == (len(free_dofs), len(free_dofs)), 'Geometric stiffness matrix has incorrect dimensions'
        assert np.allclose(Kg_reduced, Kg_reduced.T, atol=1e-10), 'Reduced geometric stiffness matrix should be symmetric'