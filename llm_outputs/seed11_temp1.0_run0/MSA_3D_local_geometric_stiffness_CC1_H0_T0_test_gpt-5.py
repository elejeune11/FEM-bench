def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.5
    A = 0.02
    I_rho = 0.0001
    K0 = fcn(L, A, I_rho, Fx2=0.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12)
    assert np.allclose(K0, np.zeros((12, 12)), atol=1e-12)
    P = 7.5
    K_ten = fcn(L, A, I_rho, Fx2=+P, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K_com = fcn(L, A, I_rho, Fx2=-P, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K_ten, K_ten.T, atol=1e-12)
    assert np.allclose(K_com, K_com.T, atol=1e-12)
    assert K_ten[7, 7] > 0.0
    assert K_com[7, 7] < 0.0
    assert K_ten[8, 8] > 0.0
    assert K_com[8, 8] < 0.0
    K_tors = fcn(L, A, I_rho, Fx2=0.0, Mx2=3.3, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert not np.allclose(K_tors, K0, atol=1e-14)
    assert np.allclose(K_tors, K_tors.T, atol=1e-12)
    K_bend = fcn(L, A, I_rho, Fx2=0.0, Mx2=0.0, My1=2.1, Mz1=-1.7, My2=-0.9, Mz2=1.3)
    assert not np.allclose(K_bend, K0, atol=1e-14)
    assert np.allclose(K_bend, K_bend.T, atol=1e-12)
    assert not np.allclose(K_tors, K_bend, atol=1e-14)
    Fx = 1.234
    K1 = fcn(L, A, I_rho, Fx2=Fx, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    K2 = fcn(L, A, I_rho, Fx2=2.0 * Fx, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    assert np.allclose(K2, 2.0 * K1, rtol=1e-12, atol=1e-12)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load; allow tolerance for single-element discretization error.
    """
    L = 3.0
    A = 0.01
    J = 5e-05
    Iz = 8e-06
    E = 210000000000.0
    Pcr_analytical = np.pi ** 2 * E * Iz / (4.0 * L ** 2)
    Ke_bz_free = E * Iz / L ** 3 * np.array([[12.0, -6.0 * L], [-6.0 * L, 4.0 * L ** 2]])
    Kg_unit_full = fcn(L, A, J, Fx2=-1.0, Mx2=0.0, My1=0.0, Mz1=0.0, My2=0.0, Mz2=0.0)
    idx = [7, 11]
    Kg_base_bz_free = -Kg_unit_full[np.ix_(idx, idx)]
    S = np.linalg.solve(Kg_base_bz_free, Ke_bz_free)
    eigvals = np.linalg.eigvals(S)
    eigvals_real = np.real(eigvals[np.isreal(eigvals)])
    eigvals_pos = eigvals_real[eigvals_real > 0.0]
    assert eigvals_pos.size >= 1
    Pcr_numerical = float(np.min(eigvals_pos))
    rel_err = abs(Pcr_numerical - Pcr_analytical) / Pcr_analytical
    assert rel_err < 0.2