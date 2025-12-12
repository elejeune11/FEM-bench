def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    x_elem = np.array([0.0, 2.0])
    E = 210000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    expected_K = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    tol = 1e-10
    K_1pt = fcn(x_elem, E, A, n_gauss=1)
    K_2pt = fcn(x_elem, E, A, n_gauss=2)
    K_3pt = fcn(x_elem, E, A, n_gauss=3)
    assert np.allclose(K_2pt, expected_K, rtol=tol, atol=tol), 'Stiffness matrix does not match analytical solution'
    assert K_2pt.shape == (2, 2), f'Stiffness matrix shape should be (2, 2), got {K_2pt.shape}'
    assert np.allclose(K_2pt, K_2pt.T, rtol=tol, atol=tol), 'Stiffness matrix is not symmetric'
    assert np.allclose(K_1pt, K_1pt.T, rtol=tol, atol=tol), 'Stiffness matrix (1-pt) is not symmetric'
    assert np.allclose(K_3pt, K_3pt.T, rtol=tol, atol=tol), 'Stiffness matrix (3-pt) is not symmetric'
    det_K = np.linalg.det(K_2pt)
    assert np.abs(det_K) < tol * (E * A / L) ** 2, f'Stiffness matrix should be singular, but determinant is {det_K}'
    assert np.allclose(K_1pt, K_2pt, rtol=tol, atol=tol), '1-point and 2-point Gauss quadrature give different results'
    assert np.allclose(K_2pt, K_3pt, rtol=tol, atol=tol), '2-point and 3-point Gauss quadrature give different results'
    assert np.allclose(K_1pt, expected_K, rtol=tol, atol=tol), '1-point Gauss quadrature does not match analytical solution'
    assert np.allclose(K_3pt, expected_K, rtol=tol, atol=tol), '3-point Gauss quadrature does not match analytical solution'
    x_elem_2 = np.array([1.0, 4.0])
    L_2 = x_elem_2[1] - x_elem_2[0]
    expected_K_2 = E * A / L_2 * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_2pt_2 = fcn(x_elem_2, E, A, n_gauss=2)
    assert np.allclose(K_2pt_2, expected_K_2, rtol=tol, atol=tol), 'Stiffness matrix incorrect for different element coordinates'
    E_2 = 70000000000.0
    A_2 = 0.005
    expected_K_3 = E_2 * A_2 / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K_2pt_3 = fcn(x_elem, E_2, A_2, n_gauss=2)
    assert np.allclose(K_2pt_3, expected_K_3, rtol=tol, atol=tol), 'Stiffness matrix incorrect for different material properties'