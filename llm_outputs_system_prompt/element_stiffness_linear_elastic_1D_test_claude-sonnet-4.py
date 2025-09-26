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
    x_unit = np.array([0.0, 1.0])
    K_unit = fcn(x_unit, E, A, 2)
    K_unit_analytical = E * A * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_unit, K_unit_analytical, rtol=1e-12), 'Unit element stiffness incorrect'
    x_offset = np.array([5.0, 7.0])
    K_offset = fcn(x_offset, E, A, 2)
    L_offset = 2.0
    K_offset_analytical = E * A / L_offset * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_offset, K_offset_analytical, rtol=1e-12), 'Offset element stiffness incorrect'
    E_small = 1.0
    A_small = 1.0
    K_small = fcn(x_elem, E_small, A_small, 2)
    K_small_analytical = E_small * A_small / L * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_small, K_small_analytical, rtol=1e-12), 'Small material properties test failed'