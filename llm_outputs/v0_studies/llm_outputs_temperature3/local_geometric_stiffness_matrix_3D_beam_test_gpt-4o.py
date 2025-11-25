def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    matrix = fcn(L, A, I_rho, 0, 0, 0, 0, 0, 0)
    assert matrix.shape == (12, 12), 'Matrix shape is incorrect'
    assert np.allclose(matrix, matrix.T), 'Matrix is not symmetric'
    assert np.allclose(matrix, np.zeros((12, 12))), 'Matrix is not zero when all loads are zero'
    matrix_tension = fcn(L, A, I_rho, 1000, 0, 0, 0, 0, 0)
    matrix_compression = fcn(L, A, I_rho, -1000, 0, 0, 0, 0, 0)
    assert np.all(matrix_tension > matrix), 'Tension does not lead to stiffening'
    assert np.all(matrix_compression < matrix), 'Compression does not lead to softening'
    matrix_torsion = fcn(L, A, I_rho, 0, 100, 0, 0, 0, 0)
    matrix_bending = fcn(L, A, I_rho, 0, 0, 100, 0, 0, 0)
    assert not np.allclose(matrix, matrix_torsion), 'Matrix does not change with torsional moment'
    assert not np.allclose(matrix, matrix_bending), 'Matrix does not change with bending moment'
    matrix_double_fx2 = fcn(L, A, I_rho, 2000, 0, 0, 0, 0, 0)
    assert np.allclose(matrix_double_fx2, 2 * matrix_tension), 'Matrix does not scale linearly with Fx2'

def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    (L, A, I_rho) = (10.0, 1.0, 1.0)
    E = 210000000000.0
    I = 1e-06
    analytical_buckling_load = np.pi ** 2 * E * I / L ** 2
    matrix_compression = fcn(L, A, I_rho, -analytical_buckling_load, 0, 0, 0, 0, 0)
    eigenvalues = np.linalg.eigvals(matrix_compression)
    critical_load = min(eigenvalues.real)
    tolerance = 0.05 * analytical_buckling_load
    assert abs(critical_load - analytical_buckling_load) < tolerance, 'Buckling load does not match analytical result'