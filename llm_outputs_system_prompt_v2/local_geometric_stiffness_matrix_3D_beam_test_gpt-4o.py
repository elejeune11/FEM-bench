def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    zero_matrix = np.zeros((12, 12))
    result = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert result.shape == (12, 12)
    assert np.allclose(result, zero_matrix)
    Fx2 = 100.0
    result = fcn(L, A, I_rho, Fx2, 0, 0, 0, 0, 0)
    assert np.allclose(result, result.T)
    result_tension = fcn(L, A, I_rho, Fx2, 0, 0, 0, 0, 0)
    result_compression = fcn(L, A, I_rho, -Fx2, 0, 0, 0, 0, 0)
    assert np.linalg.norm(result_tension) > np.linalg.norm(result_compression)
    (Mx2, My1, Mz1, My2, Mz2) = (10.0, 5.0, 5.0, 5.0, 5.0)
    result_moments = fcn(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    assert not np.allclose(result_tension, result_moments)
    result_scaled = fcn(L, A, I_rho, 2 * Fx2, 0, 0, 0, 0, 0)
    assert np.allclose(result_scaled, 2 * result_tension)

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    E = 210000000000.0
    I = 1.0
    analytical_buckling_load = np.pi ** 2 * E * I / L ** 2
    Fx2 = -analytical_buckling_load * 0.99
    result = fcn(L, A, I_rho, Fx2, 0, 0, 0, 0, 0)
    eigenvalues = np.linalg.eigvals(result)
    assert np.any(np.isclose(eigenvalues, 0, atol=1e-05))
    assert np.isclose(Fx2, -analytical_buckling_load, rtol=0.01)