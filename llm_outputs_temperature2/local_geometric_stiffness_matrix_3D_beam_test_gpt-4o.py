def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    matrix_zero_loads = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert matrix_zero_loads.shape == (12, 12)
    assert np.allclose(matrix_zero_loads, np.zeros((12, 12)))
    assert np.allclose(matrix_zero_loads, matrix_zero_loads.T)
    Fx2_tension = 100.0
    matrix_tension = fcn(L, A, I_rho, Fx2_tension, Mx2, My1, Mz1, My2, Mz2)
    assert np.all(matrix_tension >= matrix_zero_loads)
    Fx2_compression = -100.0
    matrix_compression = fcn(L, A, I_rho, Fx2_compression, Mx2, My1, Mz1, My2, Mz2)
    assert np.all(matrix_compression <= matrix_zero_loads)
    Mx2_varied = 50.0
    matrix_varied_moments = fcn(L, A, I_rho, Fx2, Mx2_varied, My1, Mz1, My2, Mz2)
    assert not np.allclose(matrix_zero_loads, matrix_varied_moments)
    Fx2_scaled = 200.0
    matrix_scaled = fcn(L, A, I_rho, Fx2_scaled, Mx2, My1, Mz1, My2, Mz2)
    assert np.allclose(matrix_scaled, 2 * matrix_tension)

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
    (Fx2, Mx2, My1, Mz1, My2, Mz2) = (-analytical_buckling_load, 0.0, 0.0, 0.0, 0.0, 0.0)
    matrix_buckling = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    eigenvalues = np.linalg.eigvals(matrix_buckling)
    assert np.any(np.isclose(eigenvalues, 0, atol=1e-05))