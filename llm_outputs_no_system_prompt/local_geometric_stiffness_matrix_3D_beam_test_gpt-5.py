def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.5
    A = 0.01
    I_rho = 0.0001
    K0 = fcn(L, A, I_rho, Fx2=0.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-14)
    Kgen = fcn(L, A, I_rho, Fx2=5.0, Mx2=1.0, My1=0.5, Mz1=-0.3, My2=0.2, Mz2=-0.1)
    assert Kgen.shape == (12, 12)
    assert np.allclose(Kgen, Kgen.T, atol=1e-10)
    dof_v_theta = [1, 5, 7, 11]
    ff_idx = [2, 3]
    Kg_tension = fcn(L, A, I_rho, Fx2=+10.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kg_compress = fcn(L, A, I_rho, Fx2=-10.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    Kt_sub = Kg_tension[np.ix_(dof_v_theta, dof_v_theta)]
    Kc_sub = Kg_compress[np.ix_(dof_v_theta, dof_v_theta)]
    Kt_ff = (Kt_sub[np.ix_(ff_idx, ff_idx)] + Kt_sub[np.ix_(ff_idx, ff_idx)].T) * 0.5
    Kc_ff = (Kc_sub[np.ix_(ff_idx, ff_idx)] + Kc_sub[np.ix_(ff_idx, ff_idx)].T) * 0.5
    evals_t = np.linalg.eigvalsh(Kt_ff)
    evals_c = np.linalg.eigvalsh(Kc_ff)
    tol_sign = 1e-12
    assert np.min(evals_t) >= -tol_sign
    assert np.max(evals_c) <= tol_sign
    K_base = fcn(L, A, I_rho, Fx2=3.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_torsion = fcn(L, A, I_rho, Fx2=3.0, Mx2=2.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_bending = fcn(L, A, I_rho, Fx2=3.0, Mx2=0.0, My1=1.0, Mz1=0.0, My2=0.0, Mz2=-1.2)
    assert not np.allclose(K_base, K_torsion, atol=1e-12, rtol=1e-12)
    assert not np.allclose(K_base, K_bending, atol=1e-12, rtol=1e-12)
    K1 = fcn(L, A, I_rho, Fx2=1.1, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K2 = fcn(L, A, I_rho, Fx2=2.2, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K2, 2.0 * K1, atol=1e-11, rtol=1e-11)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct
    buckling load for a cantilever column. Compare numerical result with the analytical
    Euler buckling load with tolerances that account for discretization error.
    """
    L = 1.0
    E = 1.0
    I = 1.0
    A = 1.0
    I_rho = 1.0
    EI = E * I
    L2 = L * L
    L3 = L2 * L
    Kb = EI / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L2, -6.0 * L, 2.0 * L2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L2, -6.0 * L, 4.0 * L2]])
    Ke = np.zeros((12, 12))
    dof_v_theta = [1, 5, 7, 11]
    for i in range(4):
        for j in range(4):
            Ke[dof_v_theta[i], dof_v_theta[j]] = Kb[i, j]
    Kg = fcn(L, A, I_rho, Fx2=-1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    free = [7, 11]
    Ke_ff = Ke[np.ix_(free, free)]
    Kg_ff = Kg[np.ix_(free, free)]
    Ke_ff = (Ke_ff + Ke_ff.T) * 0.5
    Kg_ff = (Kg_ff + Kg_ff.T) * 0.5
    M = -np.linalg.solve(Kg_ff, Ke_ff)
    eigvals = np.linalg.eigvals(M)
    eigvals = np.real_if_close(eigvals)
    positive = [val.real for val in eigvals if val.real > 0]
    assert len(positive) >= 1
    P_cr_num = min(positive)
    P_cr_analytical = np.pi ** 2 * E * I / (4.0 * L * L)
    rel_err = abs(P_cr_num - P_cr_analytical) / P_cr_analytical
    assert rel_err <= 0.1