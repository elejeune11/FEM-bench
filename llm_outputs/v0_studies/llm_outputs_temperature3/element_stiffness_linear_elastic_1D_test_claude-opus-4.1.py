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
    x_elem = np.array([2.0, 7.0])
    E = 200000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    k_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    k_2gauss = fcn(x_elem, E, A, 2)
    assert k_2gauss.shape == (2, 2)
    assert np.allclose(k_2gauss, k_2gauss.T)
    assert np.allclose(k_2gauss, k_analytical, rtol=1e-10, atol=1e-10)
    det = np.linalg.det(k_2gauss)
    assert np.abs(det) < 1e-06
    k_1gauss = fcn(x_elem, E, A, 1)
    k_3gauss = fcn(x_elem, E, A, 3)
    assert np.allclose(k_1gauss, k_2gauss, rtol=1e-10, atol=1e-10)
    assert np.allclose(k_2gauss, k_3gauss, rtol=1e-10, atol=1e-10)
    x_elem2 = np.array([0.0, 1.0])
    E2 = 1000000.0
    A2 = 0.5
    L2 = 1.0
    k_expected2 = E2 * A2 / L2 * np.array([[1, -1], [-1, 1]])
    k_computed2 = fcn(x_elem2, E2, A2, 2)
    assert np.allclose(k_computed2, k_expected2, rtol=1e-10, atol=1e-10)
    x_elem3 = np.array([-5.0, -2.0])
    E3 = 70000000000.0
    A3 = 0.002
    L3 = 3.0
    k_expected3 = E3 * A3 / L3 * np.array([[1, -1], [-1, 1]])
    k_computed3 = fcn(x_elem3, E3, A3, 2)
    assert np.allclose(k_computed3, k_expected3, rtol=1e-10, atol=1e-10)
    x_elem4 = np.array([0.0, 1e-06])
    E4 = 1.0
    A4 = 1.0
    L4 = 1e-06
    k_expected4 = E4 * A4 / L4 * np.array([[1, -1], [-1, 1]])
    k_computed4 = fcn(x_elem4, E4, A4, 2)
    assert np.allclose(k_computed4, k_expected4, rtol=1e-10, atol=1e-10)