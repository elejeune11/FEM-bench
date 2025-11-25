def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    K_zero = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert K_zero.shape == (12, 12)
    assert np.allclose(K_zero, np.zeros((12, 12)))
    assert np.allclose(K_zero, K_zero.T)
    Fx2_tension = 100.0
    K_tension = fcn(L, A, I_rho, Fx2_tension, Mx2, My1, Mz1, My2, Mz2)
    assert np.all(K_tension >= K_zero)
    Fx2_compression = -100.0
    K_compression = fcn(L, A, I_rho, Fx2_compression, Mx2, My1, Mz1, My2, Mz2)
    assert np.all(K_compression <= K_zero)
    Mx2_varied = 50.0
    My1_varied = 50.0
    K_varied = fcn(L, A, I_rho, Fx2, Mx2_varied, My1_varied, Mz1, My2, Mz2)
    assert not np.allclose(K_varied, K_zero)
    Fx2_scaled = 200.0
    K_scaled = fcn(L, A, I_rho, Fx2_scaled, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_scaled, 2 * K_tension)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    E = 210000000000.0
    I = 0.0001
    analytical_buckling_load = np.pi ** 2 * E * I / L ** 2
    Fx2 = -analytical_buckling_load * 0.99
    (Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0)
    K_g = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    eigenvalues = np.linalg.eigvals(K_g)
    min_eigenvalue = np.min(np.abs(eigenvalues))
    assert min_eigenvalue < 1e-05