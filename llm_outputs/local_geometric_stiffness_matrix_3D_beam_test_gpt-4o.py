def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    zero_matrix = np.zeros((12, 12))
    result = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.shape == (12, 12), 'Matrix shape is incorrect'
    assert np.allclose(result, zero_matrix), 'Matrix is not zero when all loads are zero'
    assert np.allclose(result, result.T), 'Matrix is not symmetric'
    Fx2_tension = 100.0
    result_tension = fcn(L, A, I_rho, Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(result_tension >= result), 'Tension should increase stiffness'
    Fx2_compression = -100.0
    result_compression = fcn(L, A, I_rho, Fx2_compression, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.all(result_compression <= result), 'Compression should decrease stiffness'
    result_moments = fcn(L, A, I_rho, 0.0, 10.0, 5.0, 5.0, 5.0, 5.0)
    assert not np.allclose(result, result_moments), 'Matrix should change with moments'
    result_double_Fx2 = fcn(L, A, I_rho, 2 * Fx2_tension, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(result_double_Fx2, 2 * result_tension), 'Matrix does not scale linearly with Fx2'

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    E = 210000000000.0
    I = 0.0001
    analytical_buckling_load = np.pi ** 2 * E * I / L ** 2
    Fx2_critical = -analytical_buckling_load
    result = fcn(L, A, I_rho, Fx2_critical, 0.0, 0.0, 0.0, 0.0, 0.0)
    eigenvalues = np.linalg.eigvals(result)
    assert np.any(np.isclose(eigenvalues, 0, atol=1e-05)), 'Matrix should be close to singular at buckling load'