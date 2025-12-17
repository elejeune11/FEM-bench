def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 3.2
    A = 0.015
    I_rho = 2.5e-05
    K0 = np.asarray(fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, rtol=0, atol=1e-12)
    assert np.allclose(K0, 0.0, rtol=0, atol=1e-14)
    F = 1234.0
    Kt = np.asarray(fcn(L, A, I_rho, F, 0.0, 0.0, 0.0, 0.0, 0.0))
    Kc = np.asarray(fcn(L, A, I_rho, -F, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.allclose(Kt, Kt.T, rtol=0, atol=1e-12)
    assert np.allclose(Kc, Kc.T, rtol=0, atol=1e-12)
    assert np.allclose(Kc, -Kt, rtol=1e-12, atol=1e-12)
    free_idx = np.array([7, 11])
    Kt_ff = Kt[np.ix_(free_idx, free_idx)]
    Kc_ff = Kc[np.ix_(free_idx, free_idx)]
    ev_t = np.linalg.eigvalsh((Kt_ff + Kt_ff.T) / 2.0)
    ev_c = np.linalg.eigvalsh((Kc_ff + Kc_ff.T) / 2.0)
    assert np.all(ev_t > 0.0)
    assert np.all(ev_c < 0.0)
    K_axial = Kt
    K_with_Mx = np.asarray(fcn(L, A, I_rho, F, 50.0, 0.0, 0.0, 0.0, 0.0))
    K_with_My = np.asarray(fcn(L, A, I_rho, F, 0.0, 25.0, 0.0, 0.0, 0.0))
    K_with_Mz = np.asarray(fcn(L, A, I_rho, F, 0.0, 0.0, 0.0, 40.0, 0.0))
    assert not np.allclose(K_with_Mx, K_axial, rtol=1e-13, atol=1e-13)
    assert not np.allclose(K_with_My, K_axial, rtol=1e-13, atol=1e-13)
    assert not np.allclose(K_with_Mz, K_axial, rtol=1e-13, atol=1e-13)
    F1 = 789.0
    alpha = 2.75
    K1 = np.asarray(fcn(L, A, I_rho, F1, 0.0, 0.0, 0.0, 0.0, 0.0))
    K2 = np.asarray(fcn(L, A, I_rho, alpha * F1, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.allclose(K2, alpha * K1, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    L = 2.4
    E = 210000000000.0
    I = 8e-06
    A = 0.02
    I_rho = 1e-05
    Kg_tension = np.asarray(fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    idx = np.array([1, 5, 7, 11])
    Kg_sub = Kg_tension[np.ix_(idx, idx)]
    Kg_ff = Kg_sub[2:, 2:]
    Ke_sub = E * I / L ** 3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]])
    Ke_ff = Ke_sub[2:, 2:]
    evals = np.linalg.eigvals(np.linalg.solve(Kg_ff, Ke_ff))
    evals = np.real(evals[np.isreal(evals)])
    evals = evals[evals > 0.0]
    Pcr_num = float(np.min(evals))
    Pcr_analytical = math.pi ** 2 * E * I / (4.0 * L ** 2)
    rel_err = abs(Pcr_num - Pcr_analytical) / Pcr_analytical
    assert rel_err < 0.03