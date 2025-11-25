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
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic."""
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T)
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical)
    assert np.abs(np.linalg.det(K)) < 1e-10
    x_elem = np.array([2.0, 5.0])
    E = 200000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical)
    x_elem = np.array([0.0, 2.5])
    E = 100.0
    A = 0.5
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert np.allclose(K1, K2, rtol=1e-10)
    assert np.allclose(K2, K3, rtol=1e-10)
    x_elem = np.array([-3.0, -1.0])
    E = 50.0
    A = 2.0
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical)
    x_elem = np.array([0.0, 1e-06])
    E = 1000000.0
    A = 0.001
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical, rtol=1e-08)