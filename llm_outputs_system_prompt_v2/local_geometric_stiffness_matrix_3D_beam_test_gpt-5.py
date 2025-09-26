def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (2.5, 1.0, 0.9)
    K_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K_zero, np.ndarray)
    assert K_zero.shape == (12, 12)
    assert np.allclose(K_zero, K_zero.T, atol=1e-12, rtol=1e-12)
    assert np.allclose(K_zero, 0.0, atol=1e-14, rtol=0.0)
    K_gen = fcn(L, A, I_rho, 10.0, 5.0, 2.0, -3.0, 4.0, -2.0)
    assert K_gen.shape == (12, 12)
    assert np.allclose(K_gen, K_gen.T, atol=1e-10, rtol=1e-10)
    idx = [1, 5, 7, 11]
    P = 1000.0
    K_tension = fcn(L, A, I_rho, P, 0.0, 0.0, 0.0, 0.0, 0.0)[np.ix_(idx, idx)]
    K_compression = fcn(L, A, I_rho, -P, 0.0, 0.0, 0.0, 0.0, 0.0)[np.ix_(idx, idx)]
    eig_t = np.linalg.eigvalsh((K_tension + K_tension.T) * 0.5)
    eig_c = np.linalg.eigvalsh((K_compression + K_compression.T) * 0.5)
    assert np.all(eig_t >= -1e-10)
    assert np.all(eig_c <= 1e-10)
    K_m0 = fcn(L, A, I_rho, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_m1 = fcn(L, A, I_rho, 50.0, 7.5, 3.2, -1.7, -2.4, 5.1)
    assert not np.allclose(K_m0, K_m1, atol=1e-12, rtol=1e-12)
    alpha = 3.7
    K1 = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    Kalpha = fcn(L, A, I_rho, alpha, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(Kalpha, alpha * K1, atol=1e-10, rtol=1e-10)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Discretization error is accounted for via a relative tolerance.
    """
    L = 1.8
    EI = 3.0
    A = 1.0
    I_rho = 0.5
    L2 = L * L
    L3 = L2 * L
    c1 = 12.0 * EI / L3
    c2 = 6.0 * EI / L2
    c3 = 4.0 * EI / L
    c4 = 2.0 * EI / L
    Ke_b = np.array([[c1, c2, -c1, c2], [c2, c3, -c2, c4], [-c1, -c2, c1, -c2], [c2, c4, -c2, c3]])
    K_full = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx = [1, 5, 7, 11]
    Kg_b = K_full[np.ix_(idx, idx)]
    free = [2, 3]
    Ke_red = Ke_b[np.ix_(free, free)]
    Kg_red = Kg_b[np.ix_(free, free)]
    M = np.linalg.solve(Kg_red, Ke_red)
    evals = np.linalg.eigvals(M)
    evals = np.real(evals[np.isreal(evals)])
    evals = evals[evals > 1e-12]
    assert evals.size >= 1
    Pcr_num = np.min(evals)
    Pcr_analytical = np.pi ** 2 * EI / (4.0 * L2)
    rel_err = abs(Pcr_num - Pcr_analytical) / Pcr_analytical
    assert Pcr_num > 0.0
    assert rel_err < 0.1