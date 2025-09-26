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
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    K = fcn(x_elem, E, A, 2)
    assert K.shape == (2, 2)
    L = 1.0
    K_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(K, K_analytical, rtol=1e-12)
    np.testing.assert_allclose(K, K.T, rtol=1e-12)
    assert abs(np.linalg.det(K)) < 1e-12
    x_elem = np.array([1.0, 3.0])
    E = 200000000000.0
    A = 0.01
    L = 2.0
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(K, K_analytical, rtol=1e-12)
    x_elem = np.array([0.0, 2.0])
    E = 1000000.0
    A = 0.1
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    np.testing.assert_allclose(K1, K2, rtol=1e-10)
    np.testing.assert_allclose(K2, K3, rtol=1e-10)
    x_elem = np.array([-2.0, 1.0])
    E = 100.0
    A = 2.0
    L = 3.0
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(K, K_analytical, rtol=1e-12)
    x_elem = np.array([0.0, 1e-06])
    E = 1.0
    A = 1.0
    L = 1e-06
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    np.testing.assert_allclose(K, K_analytical, rtol=1e-10)
    assert np.allclose(K1, K1.T)
    assert np.allclose(K2, K2.T)
    assert np.allclose(K3, K3.T)
    assert abs(np.linalg.det(K1)) < 1e-10
    assert abs(np.linalg.det(K2)) < 1e-10
    assert abs(np.linalg.det(K3)) < 1e-10