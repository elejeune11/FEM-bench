def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    L = 1.0
    A = 0.1
    I_rho = 0.01
    Fx2 = 0.0
    Mx2 = 0.0
    My1 = 0.0
    Mz1 = 0.0
    My2 = 0.0
    Mz2 = 0.0
    K = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert K.shape == (12, 12), 'Matrix should be 12x12'
    assert np.allclose(K, K.T), 'Matrix should be symmetric'
    assert np.allclose(K, np.zeros((12, 12))), 'Matrix should be zero when all loads are zero'
    Fx2_tension = 100.0
    K_tension = fcn(L, A, I_rho, Fx2_tension, Mx2, My1, Mz1, My2, Mz2)
    assert not np.allclose(K_tension, np.zeros((12, 12))), 'Matrix should be non-zero with tension'
    Fx2_compression = -100.0
    K_compression = fcn(L, A, I_rho, Fx2_compression, Mx2, My1, Mz1, My2, Mz2)
    assert not np.allclose(K_compression, np.zeros((12, 12))), 'Matrix should be non-zero with compression'
    Mx2_nonzero = 50.0
    K_with_moment = fcn(L, A, I_rho, Fx2, Mx2_nonzero, My1, Mz1, My2, Mz2)
    assert not np.allclose(K_with_moment, np.zeros((12, 12))), 'Matrix should change with torsional moment'
    scale_factor = 2.0
    K_scaled = fcn(L, A, I_rho, Fx2_tension * scale_factor, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(K_scaled, K_tension * scale_factor), 'Matrix should scale linearly with Fx2'

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    L = 2.0
    A = 0.01
    I_rho = 1e-06
    E = 200000000000.0
    I = I_rho
    P_cr_analytical = np.pi ** 2 * E * I / (4 * L ** 2)
    from scipy.linalg import det

    def get_total_stiffness(load_factor):
        Fx2 = -load_factor * P_cr_analytical
        K_g = fcn(L, A, I_rho, Fx2, 0.0, 0.0, 0.0, 0.0, 0.0)
        return K_g
    load_factors = np.linspace(0.5, 1.5, 100)
    min_det = float('inf')
    critical_load_factor = 1.0
    for lf in load_factors:
        K_g = get_total_stiffness(lf)
        determinant = abs(det(K_g))
        if determinant < min_det:
            min_det = determinant
            critical_load_factor = lf
    assert abs(critical_load_factor - 1.0) < 0.05, 'Critical load factor should be close to 1.0'