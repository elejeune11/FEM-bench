import numpy as np

def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.

    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:

    1. Analytical correctness:
        - For an element of length L with modulus E and area A,
          the stiffness matrix should match: (EA/L) * [[1, -1], [-1, 1]]

    2. Shape and symmetry:
        - The stiffness matrix must be 2x2 and symmetric.

    3. Singularity:
        - The matrix should be singular (zero determinant) for an unconstrained single element,
          reflecting rigid body motion.

    4. Integration consistency:
        - The result must be identical (within numerical tolerance) for 1-, 2-, and 3-point
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    E = 210e9
    A = 3e-4
    x_elem = np.array([2.0, 5.0])
    L = x_elem[1] - x_elem[0]

    # Analytical solution
    k_analytical = (E * A / L) * np.array([[1, -1], [-1, 1]])

    # Check result for default Gauss points (assume 2 is default)
    k_2 = fcn(x_elem, E, A, 2)

    # 1. Analytical correctness
    assert np.allclose(k_2, k_analytical, rtol=1e-12, atol=1e-12), "Stiffness does not match analytical solution"

    # 2. Shape and symmetry
    assert k_2.shape == (2,2), "Stiffness matrix must be 2x2"
    assert np.allclose(k_2, k_2.T, rtol=1e-15, atol=1e-15), "Stiffness matrix must be symmetric"

    # 3. Singularity
    det = np.linalg.det(k_2)
    assert abs(det) < 1e-12, "Stiffness matrix should be singular for a free bar (zero determinant)"

    # 4. Integration consistency: 1, 2, 3 Gauss points must produce identical result
    k_1 = fcn(x_elem, E, A, 1)
    k_3 = fcn(x_elem, E, A, 3)
    assert np.allclose(k_1, k_2, rtol=1e-12, atol=1e-12), "Integration with 1 and 2 Gauss points must agree"
    assert np.allclose(k_3, k_2, rtol=1e-12, atol=1e-12), "Integration with 3 and 2 Gauss points must agree"