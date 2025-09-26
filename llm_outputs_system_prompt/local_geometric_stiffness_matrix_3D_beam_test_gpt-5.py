def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 2.0
    A = 1.0
    I_rho = 1.0
    K0 = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (12, 12)
    assert np.allclose(K0, K0.T, atol=1e-12, rtol=0.0)
    assert np.allclose(K0, np.zeros_like(K0), atol=1e-14, rtol=0.0)
    Fx = 10.0
    K_tension = fcn(L, A, I_rho, Fx, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_compression = fcn(L, A, I_rho, -Fx, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_tension.shape == (12, 12)
    assert np.allclose(K_tension, K_tension.T, atol=1e-12, rtol=0.0)
    assert np.linalg.norm(K_tension) > 0.0
    assert np.allclose(K_tension, -K_compression, atol=1e-12, rtol=0.0)
    for idx in (1, 2, 7, 8):
        assert K_tension[idx, idx] > 0.0
        assert K_compression[idx, idx] < 0.0
    K_torsion = fcn(L, A, I_rho, Fx, 5.0, 0.0, 0.0, 0.0, 0.0)
    K_bending = fcn(L, A, I_rho, Fx, 0.0, 2.0, -3.0, -2.0, 4.0)
    assert not np.allclose(K_tension, K_torsion, atol=1e-14, rtol=0.0)
    assert not np.allclose(K_tension, K_bending, atol=1e-14, rtol=0.0)
    assert K_torsion.shape == (12, 12)
    assert K_bending.shape == (12, 12)
    assert np.allclose(K_torsion, K_torsion.T, atol=1e-12, rtol=0.0)
    assert np.allclose(K_bending, K_bending.T, atol=1e-12, rtol=0.0)
    K_2Fx = fcn(L, A, I_rho, 2 * Fx, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_2Fx, 2.0 * K_tension, atol=1e-12, rtol=0.0)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    L = 1.0
    A = 1.0
    I_rho = 1.0
    E = 1.0
    Iy = 1.0
    L3 = L ** 3
    Ke4 = E * Iy / L3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]])
    Kg = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    idx4 = [2, 4, 8, 10]
    Kg4 = Kg[np.ix_(idx4, idx4)]
    Ke_rr = Ke4[2:, 2:]
    Kg_rr = Kg4[2:, 2:]
    M = np.linalg.solve(Kg_rr, Ke_rr)
    eigvals = np.linalg.eigvals(M)
    eigvals = np.real(eigvals[np.isreal(eigvals)])
    positive = eigvals[eigvals > 0]
    assert positive.size > 0
    P_cr_num = positive.min()
    P_cr_exact = math.pi ** 2 / 4.0 * (E * Iy) / L ** 2
    rel_err = abs(P_cr_num - P_cr_exact) / P_cr_exact
    assert rel_err < 0.05