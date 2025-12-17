def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    import numpy as np
    L = 3.7
    A = 0.015
    I_rho = 2e-05
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12, rtol=0.0)
    P = 123.456
    K_tens = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_comp = fcn(L, A, I_rho, -P, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_tens.shape == (12, 12) and K_comp.shape == (12, 12)
    assert np.allclose(K_tens, K_tens.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K_comp, K_comp.T, atol=1e-12, rtol=1e-12)
    K_2tens = fcn(L, A, I_rho, 2.0 * P, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_2tens, 2.0 * K_tens, atol=1e-10, rtol=1e-10)
    assert np.allclose(K_comp, -K_tens, atol=1e-10, rtol=1e-10)
    bending_dof = [1, 2, 7, 8]
    sum_diag_tension = float(np.sum(np.diag(K_tens)[bending_dof]))
    sum_diag_compression = float(np.sum(np.diag(K_comp)[bending_dof]))
    assert sum_diag_tension > 0.0
    assert sum_diag_compression < 0.0
    K_mx = fcn(L, A, I_rho, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0)
    K_my1 = fcn(L, A, I_rho, 0.0, 0.0, 15.0, 0.0, 0.0, 0.0)
    K_mz1 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0)
    K_my2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, -8.0, 0.0)
    K_mz2 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
    for Km in (K_mx, K_my1, K_mz1, K_my2, K_mz2):
        assert Km.shape == (12, 12)
        assert np.allclose(Km, Km.T, atol=1e-12, rtol=1e-12)
        assert not np.allclose(Km, np.zeros((12, 12)), atol=1e-12, rtol=0.0)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to
    the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    import numpy as np
    L = 2.0
    E = 210000000000.0
    Iz = 8.333e-06
    A = 0.01
    I_rho = 1e-06
    Ke_sub = E * Iz / L ** 3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]])
    Kg_full = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx = [1, 5, 7, 11]
    Kg_sub = Kg_full[np.ix_(idx, idx)]
    red = [2, 3]
    Ke_red = Ke_sub[np.ix_(red, red)]
    Kg_red = Kg_sub[np.ix_(red, red)]
    B = np.linalg.solve(Kg_red, Ke_red)
    evals = np.linalg.eigvals(B)
    evals = np.real(evals)
    evals = evals[evals > 1e-12]
    assert evals.size >= 1
    Pcr_FE = float(np.min(evals))
    Pcr_analytical = np.pi ** 2 * E * Iz / (4.0 * L ** 2)
    rel_err = abs(Pcr_FE - Pcr_analytical) / Pcr_analytical
    assert rel_err < 0.2