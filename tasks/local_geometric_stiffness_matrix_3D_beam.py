import numpy as np
import scipy


def local_geometric_stiffness_matrix_3D_beam(
    L: float,
    A: float,
    I_rho: float,
    Fx2: float,
    Mx2: float,
    My1: float,
    Mz1: float,
    My2: float,
    Mz2: float
) -> np.ndarray:
    """
    Return the 12x12 local geometric stiffness matrix with torsion-bending coupling for a 3D Euler-Bernoulli beam element.

    The beam is assumed to be aligned with the local x-axis. The geometric stiffness matrix is used in conjunction with the elastic stiffness matrix for nonlinear structural analysis.

    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Where:
        - u, v, w: displacements along local x, y, z axes
        - θx, θy, θz: rotations about local x, y, z axes
        - Subscripts 1 and 2 refer to nodes i and j of the element

    Parameters:
        L (float): Length of the beam element [length units]
        A (float): Cross-sectional area [length² units]
        I_rho (float): Polar moment of inertia about the x-axis [length⁴ units]
        Fx2 (float): Internal axial force in the element (positive = tension), evaluated at node 2 [force units]
        Mx2 (float): Torsional moment at node 2 about x-axis [forcexlength units]
        My1 (float): Bending moment at node 1 about y-axis [forcexlength units]
        Mz1 (float): Bending moment at node 1 about z-axis [forcexlength units]
        My2 (float): Bending moment at node 2 about y-axis [forcexlength units]
        Mz2 (float): Bending moment at node 2 about z-axis [forcexlength units]

    Returns:
        np.ndarray: A 12x12 symmetric geometric stiffness matrix in local coordinates.
                    Positive axial force (tension) contributes to element stiffness;
                    negative axial force (compression) can lead to instability.

    Notes:
    Effects captured
        * Axial load stiffening/softening (second order “P-Δ” and “P-θ” effects):
        + **Tension (+Fx2)** increases lateral/torsional stiffness.
        + Compression (-Fx2) decreases it and may trigger buckling when K_e + K_g becomes singular.
        * Coupling of torsion (Mx2) with bending about the local y and z axes.
        * Coupling between end bending moments (My, Mz) and both transverse displacements and rotations (“moment-displacement” and “moment-rotation”).
        * Interaction between the two element nodes (no lumping of geometric terms).

    Implementation details
        * Consistent 12 x 12 matrix for an Euler-Bernoulli beam.
        * Cross-section assumed prismatic and doubly symmetric; no shear deformation, non-uniform torsion, or Wagner warping effects are included.
        * Intended for second-order geometric nonlinear analysis, large displacement static equilibrium, and eigenvalue based buckling calculations.
        * Inputs `Fx2`, `Mx2`, `My1`, `Mz1`, `My2`, `Mz2` are normally taken from the element's internal force vector at the start of the load increment/Newton iteration.
        * Valid in a small strain, moderate rotation framework (Δ L large, ε small).
    """
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 3] = My1 / L
    k_g[1, 4] = Mx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 9] = My2 / L
    k_g[1, 10] = -Mx2 / L
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 3] = Mz1 / L
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 5] = Mx2 / L
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 9] = Mz2 / L
    k_g[2, 10] = -Fx2 / 10.0
    k_g[2, 11] = -Mx2 / L
    k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
    k_g[3, 5] = (2.0 * My1 - My2) / 6.0
    k_g[3, 7] = -My1 / L
    k_g[3, 8] = -Mz1 / L
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[3, 11] = (My1 + My2) / 6.0
    k_g[4, 7] = -Mx2 / L
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[4, 11] = Mx2 / 2.0
    k_g[5, 7] = -Fx2 / 10.0
    k_g[5, 8] = -Mx2 / L
    k_g[5, 9] = (My1 + My2) / 6.0
    k_g[5, 10] = -Mx2 / 2.0
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 9] = -My2 / L
    k_g[7, 10] = Mx2 / L
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 9] = -Mz2 / L
    k_g[8, 10] = Fx2 / 10.0
    k_g[8, 11] = Mx2 / L
    k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
    k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g


def local_geometric_stiffness_matrix_3D_beam_no_moment_terms(
    L: float,
    A: float,
    I_rho: float,
    Fx2: float,
    Mx2: float,
    My1: float,
    Mz1: float,
    My2: float,
    Mz2: float
) -> np.ndarray:
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 10] = -Fx2 / 10.0
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[5, 7] = -Fx2 / 10
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 10] = Fx2 / 10.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g


def all_random_geometric(
    L: float,
    A: float,
    I_rho: float,
    Fx2: float,
    Mx2: float,
    My1: float,
    Mz1: float,
    My2: float,
    Mz2: float
) -> np.ndarray:
    return np.random.random((12, 12))


def test_local_geometric_stiffness_matrix_3D_beam_comprehensive(fcn):
    """
    Comprehensive test for local_geometric_stiffness_matrix_3D_beam:
    - shape and symmetry
    - zero matrix when all loads are zero
    - axial force (Fx2) leads to stiffening (tension) or softening (compression)
    - matrix changes when torsional and bending moments are varied
    - matrix scales linearly with Fx2
    """
    L = 2.0
    A = 0.01
    I_rho = 5e-6

    Fx2_tens = 1000.0   # Tension
    Fx2_comp = -1000.0  # Compression
    Mx2 = 200.0
    My1, Mz1 = 50.0, -75.0
    My2, Mz2 = -25.0, 100.0

    # --- Basic structure and symmetry ---
    K = fcn(L, A, I_rho, Fx2_tens, Mx2, My1, Mz1, My2, Mz2)
    assert K.shape == (12, 12), "Matrix shape must be 12x12"
    assert np.allclose(K, K.T, atol=1e-10), "Matrix must be symmetric"

    # --- Zero matrix when all inputs are zero ---
    K_zero = fcn(L, A, I_rho, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert np.allclose(K_zero, 0.0), "Expected zero matrix when all inputs are zero"

    # --- Compression leads to softening ---
    K_comp = fcn(L, A, I_rho, Fx2_comp, Mx2, My1, Mz1, My2, Mz2)
    assert K[1, 1] > K_comp[1, 1], "Transverse stiffness should reduce under compression"
    assert K[3, 3] > K_comp[3, 3], "Torsional stiffness should reduce under compression"

    # --- Matrix scales linearly with Fx2 ---
    K_scaled = fcn(L, A, I_rho, 2 * Fx2_tens, Mx2, My1, Mz1, My2, Mz2)
    scale_factor = 2.0
    # Diagonal terms directly proportional to Fx2
    assert np.isclose(K_scaled[0, 0], scale_factor * K[0, 0], rtol=1e-10)
    assert np.isclose(K_scaled[1, 1], scale_factor * K[1, 1], rtol=1e-10)
    assert np.isclose(K_scaled[3, 3], scale_factor * K[3, 3], rtol=1e-10)

    # --- Matrix changes when moment inputs change ---
    K_no_mx2 = fcn(L, A, I_rho, Fx2_tens, 0.0, My1, Mz1, My2, Mz2)
    assert not np.isclose(K[4, 11], K_no_mx2[4, 11], atol=1e-10), "Mx2 should affect [4,11]"

    K_no_my1 = fcn(L, A, I_rho, Fx2_tens, Mx2, 0.0, Mz1, My2, Mz2)
    assert not np.isclose(K[1, 3], K_no_my1[1, 3], atol=1e-10), "My1 should affect [1,3]"

    K_no_mz1 = fcn(L, A, I_rho, Fx2_tens, Mx2, My1, 0.0, My2, Mz2)
    assert not np.isclose(K[2, 3], K_no_mz1[2, 3], atol=1e-10), "Mz1 should affect [2,3]"

    K_no_my2 = fcn(L, A, I_rho, Fx2_tens, Mx2, My1, Mz1, 0.0, Mz2)
    assert not np.isclose(K[1, 9], K_no_my2[1, 9], atol=1e-10), "My2 should affect [1,9]"

    K_no_mz2 = fcn(L, A, I_rho, Fx2_tens, Mx2, My1, Mz1, My2, 0.0)
    assert not np.isclose(K[2, 9], K_no_mz2[2, 9], atol=1e-10), "Mz2 should affect [2,9]"


def test_euler_buckling_cantilever_column(fcn):
    """
    Test the geometric stiffness matrix formulation by checking that it leads to the correct buckling load for a cantilever column.
    Compare numerical result with the analytical Euler buckling load.
    Design the test so that comparison tolerances account for discretization error.
    """
    E = 210e4        # Young's modulus [Pa]
    A = 0.01         # Cross-sectional area [m²]
    L = 2.0          # Length [m]
    Iz = 1e-2        # Moment of inertia for bending about z
    Iy = 5e-1        # not used here
    J = 1e-2         # not used here
    nu = 0.3         # Poisson’s ratio
    I_rho = 1e-3     # polar inertia, small value for numerical stability

    Fx_guess = -1.0  # apply unit axial compression

    # Moments are zero for pure axial buckling
    Mx2 = My1 = Mz1 = My2 = Mz2 = 0.0

    # Construct element stiffness matrices
    Ke = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    Ke[0, 0] = axial_stiffness
    Ke[0, 6] = -axial_stiffness
    Ke[6, 0] = -axial_stiffness
    Ke[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    Ke[3, 3] = torsional_stiffness
    Ke[3, 9] = -torsional_stiffness
    Ke[9, 3] = -torsional_stiffness
    Ke[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    Ke[1, 1] = E * 12.0 * Iz / L ** 3.0
    Ke[1, 7] = E * -12.0 * Iz / L ** 3.0
    Ke[7, 1] = E * -12.0 * Iz / L ** 3.0
    Ke[7, 7] = E * 12.0 * Iz / L ** 3.0
    Ke[1, 5] = E * 6.0 * Iz / L ** 2.0
    Ke[5, 1] = E * 6.0 * Iz / L ** 2.0
    Ke[1, 11] = E * 6.0 * Iz / L ** 2.0
    Ke[11, 1] = E * 6.0 * Iz / L ** 2.0
    Ke[5, 7] = E * -6.0 * Iz / L ** 2.0
    Ke[7, 5] = E * -6.0 * Iz / L ** 2.0
    Ke[7, 11] = E * -6.0 * Iz / L ** 2.0
    Ke[11, 7] = E * -6.0 * Iz / L ** 2.0
    Ke[5, 5] = E * 4.0 * Iz / L
    Ke[11, 11] = E * 4.0 * Iz / L
    Ke[5, 11] = E * 2.0 * Iz / L
    Ke[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    Ke[2, 2] = E * 12.0 * Iy / L ** 3.0
    Ke[2, 8] = E * -12.0 * Iy / L ** 3.0
    Ke[8, 2] = E * -12.0 * Iy / L ** 3.0
    Ke[8, 8] = E * 12.0 * Iy / L ** 3.0
    Ke[2, 4] = E * -6.0 * Iy / L ** 2.0
    Ke[4, 2] = E * -6.0 * Iy / L ** 2.0
    Ke[2, 10] = E * -6.0 * Iy / L ** 2.0
    Ke[10, 2] = E * -6.0 * Iy / L ** 2.0
    Ke[4, 8] = E * 6.0 * Iy / L ** 2.0
    Ke[8, 4] = E * 6.0 * Iy / L ** 2.0
    Ke[8, 10] = E * 6.0 * Iy / L ** 2.0
    Ke[10, 8] = E * 6.0 * Iy / L ** 2.0
    Ke[4, 4] = E * 4.0 * Iy / L
    Ke[10, 10] = E * 4.0 * Iy / L
    Ke[4, 10] = E * 2.0 * Iy / L
    Ke[10, 4] = E * 2.0 * Iy / L

    Kg = fcn(L, A, I_rho, Fx_guess, Mx2, My1, Mz1, My2, Mz2)

    # Cantilever: node 1 fixed, node 2 free → DOFs 6–11
    free_dofs = np.asarray([6, 7, 11]) #np.arange(6, 12)
    Ke_ff = Ke[np.ix_(free_dofs, free_dofs)]
    Kg_ff = Kg[np.ix_(free_dofs, free_dofs)]

    # Solve generalized eigenvalue problem
    eigvals, _ = scipy.linalg.eig(Ke_ff, -Kg_ff)
    eigvals = np.real_if_close(eigvals)
    eigvals = eigvals[np.isreal(eigvals)]
    eigvals = eigvals[np.real(eigvals) > 0]

    lambda_cr = np.min(np.real(eigvals))
    P_cr_numeric = lambda_cr  # because Fx_guess = -1.0

    # Analytical Euler buckling load for cantilever:
    P_cr_exact = (np.pi**2 * E * Iz) / (4 * L**2)

    assert np.isclose(P_cr_numeric, P_cr_exact, rtol=1e-2)


def task_info():
    task_id = "local_geometric_stiffness_matrix_3D_beam"
    task_short_description = "creates an element geometric stiffness matrix for a 3D beam"
    created_date = "2025-08-04"
    created_by = "elejeune11"
    main_fcn = local_geometric_stiffness_matrix_3D_beam
    required_imports = ["import numpy as np", "import scipy", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [
        [1.0, 1e-4, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 5e-5, 2e-7, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 2e-5, 5e-7, -300.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.2, 1.5e-4, 3e-6, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 1e-4, 2e-6, 0.0, 0.0, 20.0, -20.0, -15.0, 25.0],
        [1.5, 2.2e-4, 1.8e-6, -500.0, 25.0, 15.0, 10.0, 5.0, -5.0],
        [0.8, 4e-4, 5e-6, -5000.0, 100.0, 100.0, 80.0, 120.0, -90.0]
    ]
    test_cases = [{"test_code": test_local_geometric_stiffness_matrix_3D_beam_comprehensive, "expected_failures": [local_geometric_stiffness_matrix_3D_beam_no_moment_terms, all_random_geometric]},
                  {"test_code": test_euler_buckling_cantilever_column, "expected_failures": [all_random_geometric]}]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
