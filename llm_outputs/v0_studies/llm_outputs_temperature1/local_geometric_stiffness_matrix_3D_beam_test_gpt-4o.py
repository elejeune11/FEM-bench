def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    matrix = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert matrix.shape == (12, 12)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix, np.zeros((12, 12)))
    matrix_tension = fcn(L, A, I_rho, 100, 0, 0, 0, 0, 0)
    assert np.all(matrix_tension >= matrix)
    matrix_compression = fcn(L, A, I_rho, -100, 0, 0, 0, 0, 0)
    assert np.all(matrix_compression <= matrix)
    matrix_torsion = fcn(L, A, I_rho, 0, 50, 0, 0, 0, 0)
    assert not np.allclose(matrix, matrix_torsion)
    matrix_bending = fcn(L, A, I_rho, 0, 0, 10, 0, 10, 0)
    assert not np.allclose(matrix, matrix_bending)
    matrix_double_Fx2 = fcn(L, A, I_rho, 200, 0, 0, 0, 0, 0)
    assert np.allclose(matrix_double_Fx2, 2 * matrix_tension)

def test_euler_buckling_cantilever_column(fcn):
    """Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    E = 210000000000.0
    I = 0.0001
    euler_buckling_load = np.pi ** 2 * E * I / (4 * L ** 2)
    matrix = fcn(L, A, I_rho, -euler_buckling_load, 0, 0, 0, 0, 0)
    eigenvalues = np.linalg.eigvals(matrix)
    min_eigenvalue = np.min(np.abs(eigenvalues))
    assert min_eigenvalue < 1e-05 * euler_buckling_load