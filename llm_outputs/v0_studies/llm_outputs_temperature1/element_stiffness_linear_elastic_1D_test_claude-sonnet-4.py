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
    E = 200000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    assert K.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K, K.T, rtol=1e-14), 'Stiffness matrix must be symmetric'
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical, rtol=1e-12), 'Stiffness matrix does not match analytical solution'
    det_K = np.linalg.det(K)
    assert abs(det_K) < 1e-10, 'Stiffness matrix should be singular (det ≈ 0)'
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert np.allclose(K1, K2, rtol=1e-12), 'Results should be identical for 1 and 2 Gauss points'
    assert np.allclose(K2, K3, rtol=1e-12), 'Results should be identical for 2 and 3 Gauss points'
    x_elem2 = np.array([1.0, 5.0])
    K2_test = fcn(x_elem2, E, A, 2)
    L2 = x_elem2[1] - x_elem2[0]
    K2_analytical = E * A / L2 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K2_test, K2_analytical, rtol=1e-12), 'Failed for different element length'
    (E2, A2) = (100000000000.0, 0.005)
    K3_test = fcn(x_elem, E2, A2, 2)
    K3_analytical = E2 * A2 / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K3_test, K3_analytical, rtol=1e-12), 'Failed for different material properties'
    row_sums = np.sum(K, axis=1)
    assert np.allclose(row_sums, 0, atol=1e-12), 'Row sums should be zero for equilibrium'