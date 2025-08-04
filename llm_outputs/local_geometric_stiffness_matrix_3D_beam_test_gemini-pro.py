def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    shape and symmetry
    zero matrix when all loads are zero
    axial force (Fx2) leads to stiffening (tension) or softening (compression)
    matrix changes when torsional and bending moments are varied
    matrix scales linearly with Fx2"""
    L = 1.0
    A = 1.0
    I_rho = 1.0
    K_g_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert K_g_zero.shape == (12, 12)
    assert_allclose(K_g_zero, np.zeros((12, 12)))
    K_g = fcn(L, A, I_rho, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert_allclose(K_g, K_g.T)
    K_g_tension = fcn(L, A, I_rho, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_compression = fcn(L, A, I_rho, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(K_g_tension, K_g_compression)
    K_g_My = fcn(L, A, I_rho, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0)
    K_g_Mz = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0)
    K_g_Mx = fcn(L, A, I_rho, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    assert not np.allclose(K_g_My, np.zeros((12, 12)))
    assert not np.allclose(K_g_Mz, np.zeros((12, 12)))
    assert not np.allclose(K_g_Mx, np.zeros((12, 12)))
    K_g_1 = fcn(L, A, I_rho, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_g_2 = fcn(L, A, I_rho, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert_allclose(K_g_2, 2 * K_g_1)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error."""
    L = 1.0
    A = 1.0
    I = 1.0
    E = 1.0
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    K_e = np.array([[E * A / L, 0, 0, 0, 0, 0, -E * A / L, 0, 0, 0, 0, 0], [0, 12 * E * I / L ** 3, 0, 0, 0, 6 * E * I / L ** 2, 0, -12 * E * I / L ** 3, 0, 0, 0, 6 * E * I / L ** 2], [0, 0, 12 * E * I / L ** 3, 0, -6 * E * I / L ** 2, 0, 0, 0, -12 * E * I / L ** 3, 0, -6 * E * I / L ** 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -6 * E * I / L ** 2, 0, 4 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0, 2 * E * I / L, 0], [0, 6 * E * I / L ** 2, 0, 0, 0, 4 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0, 2 * E * I / L], [-E * A / L, 0, 0, 0, 0, 0, E * A / L, 0, 0, 0, 0, 0], [0, -12 * E * I / L ** 3, 0, 0, 0, -6 * E * I / L ** 2, 0, 12 * E * I / L ** 3, 0, 0, 0, -6 * E * I / L ** 2], [0, 0, -12 * E * I / L ** 3, 0, 6 * E * I / L ** 2, 0, 0, 0, 12 * E * I / L ** 3, 0, 6 * E * I / L ** 2, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -6 * E * I / L ** 2, 0, 2 * E * I / L, 0, 0, 0, 6 * E * I / L ** 2, 0, 4 * E * I / L, 0], [0, 6 * E * I / L ** 2, 0, 0, 0, 2 * E * I / L, 0, -6 * E * I / L ** 2, 0, 0, 0, 4 * E * I / L]])
    P = -0.99 * P_cr_analytical
    K_g = fcn(L, A, I, P, 0.0, 0.0, 0.0, 0.0, 0.0)
    K = K_e + K_g
    (w, v) = np.linalg.eig(K)
    assert np.all(np.real(w) > 0)