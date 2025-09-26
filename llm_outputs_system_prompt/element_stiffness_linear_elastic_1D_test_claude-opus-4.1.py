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
    import numpy as np
    x_elem = np.array([0.0, 1.0])
    E = 1.0
    A = 1.0
    L = x_elem[1] - x_elem[0]
    K = fcn(x_elem, E, A, 2)
    assert K.shape == (2, 2), 'Stiffness matrix must be 2x2'
    assert np.allclose(K, K.T), 'Stiffness matrix must be symmetric'
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical, rtol=1e-10, atol=1e-12), 'Stiffness matrix does not match analytical solution'
    det_K = np.linalg.det(K)
    assert np.abs(det_K) < 1e-10, 'Stiffness matrix should be singular (det ≈ 0)'
    x_elem2 = np.array([2.0, 5.0])
    L2 = x_elem2[1] - x_elem2[0]
    K2 = fcn(x_elem2, E, A, 2)
    K2_analytical = E * A / L2 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K2, K2_analytical, rtol=1e-10, atol=1e-12), 'Stiffness matrix incorrect for different element length'
    E3 = 210000000000.0
    A3 = 0.01
    x_elem3 = np.array([0.0, 0.5])
    L3 = x_elem3[1] - x_elem3[0]
    K3 = fcn(x_elem3, E3, A3, 2)
    K3_analytical = E3 * A3 / L3 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K3, K3_analytical, rtol=1e-10, atol=1e-12), 'Stiffness matrix incorrect for different material properties'
    x_elem4 = np.array([1.5, 3.7])
    E4 = 100.0
    A4 = 2.5
    K_1gauss = fcn(x_elem4, E4, A4, 1)
    K_2gauss = fcn(x_elem4, E4, A4, 2)
    K_3gauss = fcn(x_elem4, E4, A4, 3)
    assert np.allclose(K_1gauss, K_2gauss, rtol=1e-10, atol=1e-12), '1-point and 2-point Gauss integration should give same result for linear element'
    assert np.allclose(K_2gauss, K_3gauss, rtol=1e-10, atol=1e-12), '2-point and 3-point Gauss integration should give same result for linear element'
    L4 = x_elem4[1] - x_elem4[0]
    K4_analytical = E4 * A4 / L4 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_1gauss, K4_analytical, rtol=1e-10, atol=1e-12), '1-point Gauss integration does not match analytical solution'
    assert np.allclose(K_2gauss, K4_analytical, rtol=1e-10, atol=1e-12), '2-point Gauss integration does not match analytical solution'
    assert np.allclose(K_3gauss, K4_analytical, rtol=1e-10, atol=1e-12), '3-point Gauss integration does not match analytical solution'
    x_elem5 = np.array([-2.0, -0.5])
    L5 = x_elem5[1] - x_elem5[0]
    K5 = fcn(x_elem5, E, A, 2)
    K5_analytical = E * A / L5 * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K5, K5_analytical, rtol=1e-10, atol=1e-12), 'Stiffness matrix incorrect for negative coordinates'