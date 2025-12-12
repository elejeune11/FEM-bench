def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `FEM_1D_local_elastic_stiffness_CC0_H3_T1` for a two-node linear element:
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
    E = 200000000000.0
    A = 0.01
    x_elem = np.array([0.0, 1.0])
    L = np.diff(x_elem)[0]
    K_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    for n_gauss in [1, 2, 3]:
        K_computed = fcn(x_elem, E, A, n_gauss)
        assert K_computed.shape == (2, 2), f'Incorrect shape for {n_gauss}-point Gauss quadrature'
        assert np.allclose(K_computed, K_computed.T), f'Stiffness matrix is not symmetric for {n_gauss}-point Gauss quadrature'
        assert np.allclose(K_computed, K_analytical), f'Stiffness matrix does not match analytical solution for {n_gauss}-point Gauss quadrature'
        assert np.isclose(np.linalg.det(K_computed), 0), f'Stiffness matrix is not singular for {n_gauss}-point Gauss quadrature'
    K_1 = fcn(x_elem, E, A, 1)
    K_2 = fcn(x_elem, E, A, 2)
    K_3 = fcn(x_elem, E, A, 3)
    assert np.allclose(K_1, K_2)
    assert np.allclose(K_2, K_3)