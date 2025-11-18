def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    x_elem = np.array([0.0, 1.0])
    E = 210000000000.0
    A = 0.01
    L = x_elem[1] - x_elem[0]
    expected_stiffness = E * A / L * np.array([[1, -1], [-1, 1]])
    for n_gauss in [1, 2, 3]:
        stiffness_matrix = fcn(x_elem, E, A, n_gauss)
        assert np.allclose(stiffness_matrix, expected_stiffness, atol=1e-08), 'Stiffness matrix does not match expected analytical result.'
        assert stiffness_matrix.shape == (2, 2), 'Stiffness matrix is not 2x2.'
        assert np.allclose(stiffness_matrix, stiffness_matrix.T, atol=1e-08), 'Stiffness matrix is not symmetric.'
        assert np.isclose(np.linalg.det(stiffness_matrix), 0, atol=1e-08), 'Stiffness matrix is not singular for an unconstrained element.'