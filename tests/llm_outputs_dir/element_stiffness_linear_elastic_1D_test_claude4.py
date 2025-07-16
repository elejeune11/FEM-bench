import numpy as np
import pytest


def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.

    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:

    1. Analytical correctness:
        - For an element of length L with modulus E and area A, the stiffness matrix should match:
          (EA/L) * [[1, -1], [-1, 1]]

    2. Shape and symmetry:
        - The stiffness matrix must be 2x2 and symmetric.

    3. Singularity:
        - The matrix should be singular (zero determinant) for an unconstrained single element,
          reflecting rigid body motion.

    4. Integration consistency:
        - The result must be identical (within numerical tolerance) for 1-, 2-, and 3-point
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    # Test parameters
    x_elem = np.array([0.0, 2.0])
    E = 200e9  # Young's modulus in Pa
    A = 0.01   # Cross-sectional area in m^2
    L = x_elem[1] - x_elem[0]  # Element length
    
    # Compute stiffness matrix with default Gauss points
    K = fcn(x_elem, E, A, 2)
    
    # 1. Test shape
    assert K.shape == (2, 2), f"Expected shape (2, 2), got {K.shape}"
    
    # 2. Test symmetry
    assert np.allclose(K, K.T), "Stiffness matrix must be symmetric"
    
    # 3. Test analytical correctness
    K_analytical = (E * A / L) * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K, K_analytical), "Stiffness matrix does not match analytical solution"
    
    # 4. Test singularity (zero determinant for unconstrained element)
    det_K = np.linalg.det(K)
    assert abs(det_K) < 1e-10, f"Stiffness matrix should be singular, determinant = {det_K}"
    
    # 5. Test integration consistency across different Gauss point counts
    K_1gauss = fcn(x_elem, E, A, 1)
    K_2gauss = fcn(x_elem, E, A, 2)
    K_3gauss = fcn(x_elem, E, A, 3)
    
    assert np.allclose(K_1gauss, K_2gauss), "1-point and 2-point Gauss integration should give identical results"
    assert np.allclose(K_2gauss, K_3gauss), "2-point and 3-point Gauss integration should give identical results"
    
    # 6. Test with different element lengths
    x_elem_short = np.array([1.0, 1.5])
    L_short = x_elem_short[1] - x_elem_short[0]
    K_short = fcn(x_elem_short, E, A, 2)
    K_short_analytical = (E * A / L_short) * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_short, K_short_analytical), "Stiffness matrix incorrect for different element length"
    
    # 7. Test with different material properties
    E_steel = 210e9
    A_large = 0.02
    K_steel = fcn(x_elem, E_steel, A_large, 2)
    K_steel_analytical = (E_steel * A_large / L) * np.array([[1, -1], [-1, 1]])
    assert np.allclose(K_steel, K_steel_analytical), "Stiffness matrix incorrect for different material properties"
    
    # 8. Test positive definiteness properties
    eigenvals = np.linalg.eigvals(K)
    eigenvals_sorted = np.sort(eigenvals)
    assert abs(eigenvals_sorted[0]) < 1e-10, "First eigenvalue should be zero (rigid body mode)"
    assert eigenvals_sorted[1] > 0, "Second eigenvalue should be positive"
    
    # 9. Test row sum property (equilibrium condition)
    row_sums = np.sum(K, axis=1)
    assert np.allclose(row_sums, 0), "Row sums should be zero for equilibrium"
    
    # 10. Test column sum property (equilibrium condition)
    col_sums = np.sum(K, axis=0)
    assert np.allclose(col_sums, 0), "Column sums should be zero for equilibrium"